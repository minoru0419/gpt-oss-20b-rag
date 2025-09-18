import os
import traceback
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma

# ------------------------
# FastAPI アプリ本体
# ------------------------
app = FastAPI()

# CORS (フロントからアクセス可能にする)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では環境変数にする
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# ディレクトリ設定
# ------------------------
UPLOAD_DIR = Path("docs")
VECTOR_DIR = Path("chroma_db")
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

# ------------------------
# Embedding & DB
# ------------------------
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(
        persist_directory=str(VECTOR_DIR),
        collection_name="docs",
        embedding_function=embeddings,
    )
    print("[INIT] Chroma 初期化成功")
except Exception as e:
    print("[ERROR] Chroma 初期化失敗:", e)
    vectordb = None

# ------------------------
# LLM 定義
# ------------------------
small_llm = OllamaLLM(model="llama3", temperature=0.2)
big_llm = OllamaLLM(model="gpt-oss:20b", temperature=0.2)


# ------------------------
# Utility
# ------------------------
def safe_print(*args):
    try:
        print(*args, flush=True)
    except Exception:
        pass


# ------------------------
# ファイルアップロード
# ------------------------
@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = file.filename
            chunk.metadata["chunk_id"] = i

        if vectordb:
            vectordb.add_documents(chunks)
            vectordb.persist()

        safe_print(f"[UPLOAD] {file.filename} chunks={len(chunks)}")
        return {"uploaded": file.filename, "chunks": len(chunks)}

    except Exception as e:
        safe_print("[ERROR] /upload failed:", repr(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------
# 質問応答
# ------------------------
@app.post("/ask")
async def ask(question: str = Form(...)):
    t0 = time.time()
    try:
        if not vectordb:
            return JSONResponse(status_code=500, content={"error": "DB未初期化"})

        docs = []
        context = ""

        try:
            pairs = vectordb.similarity_search_with_score(question, k=3)
            if pairs:
                docs = [doc for doc, _ in pairs]
                context = "\n---\n".join([d.page_content for d in docs])
        except Exception as e:
            safe_print("[WARN] similarity_search失敗:", e)

        llm = small_llm if len(question) < 50 else big_llm

        prompt = (
            "あなたは日本語専用アシスタントです。必ず日本語で答えてください。\n\n"
            f"質問:\n{question}\n\n"
            f"参考文書:\n{context}"
        )

        answer = llm.invoke(prompt)

        elapsed = time.time() - t0
        safe_print(f"[ASK] Q={question[:30]}... time={elapsed:.2f}s")

        return {
            "answer": answer,
            "sources": [doc.metadata for doc in docs],
            "elapsed": elapsed,
        }

    except Exception as e:
        safe_print("[ERROR] /ask failed:", repr(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
