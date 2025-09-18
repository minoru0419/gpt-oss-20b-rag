from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import traceback
import time
import re
import ast
import operator as op
import os

import numpy as np
from joblib import dump, load
from sklearn.decomposition import PCA
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings

# ------------------------
# FastAPI アプリ本体
# ------------------------
app = FastAPI()

# CORS 設定（環境変数 APP_ENV に応じて切替）
APP_ENV = os.getenv("APP_ENV", "dev")
if APP_ENV == "prod":
    allow_origins = ["https://your-frontend.example.com"]
else:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# ディレクトリ
# ------------------------
UPLOAD_DIR = Path("docs")
UPLOAD_DIR.mkdir(exist_ok=True)

# PCA 関連
REDUCED_DIM = int(os.getenv("REDUCED_DIM", "256"))
USE_PCA = os.getenv("USE_PCA", "1") == "1"
PCA_FILE = Path("pca.joblib")

# ------------------------
# 埋め込みラッパー (PCA + float16)
# ------------------------
class PCAReducedEmbeddings(Embeddings):
    def __init__(self, base: Embeddings, pca: PCA | None):
        self.base = base
        self.pca = pca

    def _post(self, arr: np.ndarray) -> list[list[float]]:
        if self.pca is not None:
            arr = self.pca.transform(arr)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        arr = arr.astype(np.float16)
        return arr.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        arr = np.array(self.base.embed_documents(texts), dtype=np.float32)
        return self._post(arr)

    def embed_query(self, text: str) -> list[float]:
        arr = np.array(self.base.embed_documents([text]), dtype=np.float32)
        return self._post(arr)[0]

# ------------------------
# ベクトルDBと LLM
# ------------------------
base_embeddings = OllamaEmbeddings(model="nomic-embed-text")
pca_model = None

def train_or_load_pca():
    """docs/ 内の文書を用いて PCA を学習 or ロード"""
    global pca_model
    if not USE_PCA:
        print("[PCA] disabled")
        return None

    if PCA_FILE.exists():
        try:
            pca_model = load(PCA_FILE)
            print(f"[PCA] loaded: {PCA_FILE} (dim -> {REDUCED_DIM})")
            return pca_model
        except Exception as e:
            print("[PCA] load failed, retraining:", e)

    # docs/ 内のテキストを収集
    texts = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    for p in UPLOAD_DIR.glob("**/*"):
        if p.is_dir():
            continue
        try:
            if p.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(p))
            else:
                loader = TextLoader(str(p), encoding="utf-8")
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            texts.extend([c.page_content for c in chunks])
        except Exception as e:
            print("[PCA] skip file read error:", p, e)

    if not texts:
        print("[PCA] no documents found, fallback to no PCA")
        return None

    print(f"[PCA] training on {len(texts)} chunks")
    base_embs = np.array(base_embeddings.embed_documents(texts), dtype=np.float32)

    pca = PCA(n_components=REDUCED_DIM, svd_solver="auto", random_state=42)
    pca.fit(base_embs)

    dump(pca, PCA_FILE)
    print(f"[PCA] saved to {PCA_FILE}; dim -> {REDUCED_DIM}")
    return pca

# 起動時に PCA をロード or 学習
pca_model = train_or_load_pca()
embeddings = PCAReducedEmbeddings(base_embeddings, pca_model) if pca_model else base_embeddings

VECTOR_DIR = Path("chroma_db_reduced" if pca_model else "chroma_db")
VECTOR_DIR.mkdir(exist_ok=True)

vectordb = Chroma(
    persist_directory=str(VECTOR_DIR),
    collection_name="docs",
    embedding_function=embeddings,
)

# 小さいモデル（速い）
small_llm = OllamaLLM(
    model="llama3",
    temperature=0.2,
    num_ctx=1024,
    num_predict=512,
)

# 大きいモデル（精度重視）
big_llm = OllamaLLM(
    model="gpt-oss:20b",
    temperature=0.2,
    num_ctx=2048,
    num_predict=1024,
)

# ------------------------
# 安全な四則演算
# ------------------------
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
}
def _safe_eval_expr(expr: str):
    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.UnaryOp):
            return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp):
            return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("invalid")
    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)
def try_math_answer(question: str):
    if re.fullmatch(r"[0-9\.\s\+\-\*\/=]+", question):
        expr = question.replace("=", "").strip()
        try:
            return str(round(_safe_eval_expr(expr), 10))
        except Exception:
            return None
    return None

# ------------------------
# コード抽出＋強制整形
# ------------------------
def extract_code_blocks(text: str):
    matches = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", text)
    if matches:
        return "\n\n".join(matches).strip()
    return text

def force_python_multiline(code: str) -> str:
    code = re.sub(r"\b(for .*?:)", r"\1\n    ", code)
    code = re.sub(r"\b(if .*?:)", r"\1\n    ", code)
    code = re.sub(r"\b(while .*?:)", r"\1\n    ", code)
    code = re.sub(r":\s+for", ":\n    for", code)
    code = re.sub(r":\s+if", ":\n    if", code)
    code = re.sub(r":\s+while", ":\n    while", code)
    code = re.sub(r"(print\(.*\))", r"\1\n", code)
    lines = [line.rstrip() for line in code.splitlines()]
    return "\n".join(lines).strip()

# ------------------------
# モード判定
# ------------------------
DOC_SCORE_THRESHOLD = 0.30

def decide_mode_by_similarity(q: str, k: int = 5):
    if len(q.strip()) < 20:
        return "general", [], ""
    try:
        pairs = vectordb.similarity_search_with_score(q, k=k)
        if not pairs:
            return "general", [], ""
        top_doc, top_score = pairs[0]
        mode = "document" if top_score <= DOC_SCORE_THRESHOLD else "general"
        docs = [doc for doc, _ in pairs[:min(5, len(pairs))]]
        context = "\n---\n".join([d.page_content for d in docs])
        if mode == "general":
            return "general", [], ""
        return "document", docs, context
    except Exception as e:
        print("[WARN] similarity fallback:", e)
        return "general", [], ""

# ------------------------
# モデル選択
# ------------------------
def select_llm(question: str, mode: str, doc_count: int):
    if mode == "general" and len(question) < 50:
        return small_llm, "small"
    if mode == "document" and doc_count < 2:
        return small_llm, "small"
    return big_llm, "big"

# ------------------------
# ファイルアップロード
# ------------------------
@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = file.filename
            chunk.metadata["chunk_id"] = i

        vectordb.add_documents(chunks)

        return {"uploaded": file.filename, "chunks": len(chunks)}
    except Exception as e:
        print("[ERROR] /upload failed:", repr(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------------
# 質問応答
# ------------------------
@app.post("/ask")
async def ask(question: str = Form(...)):
    t0 = time.time()
    try:
        math_result = try_math_answer(question)
        if math_result is not None:
            return {"answer": math_result, "sources": [], "mode": "math"}

        mode, docs, context = decide_mode_by_similarity(question, k=5)
        if mode == "general":
            docs, context = [], ""

        llm, llm_size = select_llm(question, mode, len(docs))

        prompt = (
            "あなたは日本語専用アシスタントです。必ず日本語で答えてください。\n"
            "質問を無視せず必ず答えてください。\n"
            "文書に答えがある場合は文書を根拠に答えてください。\n"
            "文書に答えがなくても一般知識で答えられる場合は一般知識で答えてください。\n"
            "どちらにも答えがない場合だけ『この文書には記載がありません』と答えてください。\n"
            "もしプログラムコードを提示する場合は以下のルールを厳守してください：\n"
            "1. 出力は適切な言語名を指定したコードブロックのみとすること。\n"
            "2. コード以外の文章（解説・コメント・補足説明）は絶対に出力してはいけない。\n"
            "3. コードは必ず実行可能な完全なプログラムにすること。\n"
            "4. トップレベルのスクリプト形式で記述し、関数やクラスを勝手に作ってはいけない。\n"
            "5. for文やif文などのブロックは必ず改行とインデントを使うこと。一行にまとめてはいけない。\n\n"
            "もし参考文書と矛盾する知識がある場合は、必ず参考文書を優先してください。\n"
            "参考文書に明記されている場合は、モデル自身の知識を使ってはいけません。\n"
            f"質問:\n{question}\n\n"
            f"参考文書:\n{context}"
        )

        answer = llm.invoke(prompt)
        cleaned = extract_code_blocks(answer)
        answer = force_python_multiline(cleaned)

        sources, seen = [], set()
        for doc in (docs or []):
            meta = doc.metadata or {}
            key = (meta.get("source"), meta.get("page"), meta.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            sources.append({"source": key[0], "page": key[1], "chunk_id": key[2]})

        elapsed = time.time() - t0
        print(f"[ASK] mode={mode} model={llm_size} time={elapsed:.2f}s")
        return {"answer": answer, "sources": sources, "mode": mode, "llm": llm_size}

    except Exception as e:
        print("[ERROR] /ask failed:", repr(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
