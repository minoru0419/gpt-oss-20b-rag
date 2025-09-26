# app.py
# ======================================================================
# RAG WebUI バックエンド (FastAPI)
# - Ollama API を利用した LLM 応答生成
# - RAG (Retrieval-Augmented Generation) 簡易実装（★永続化対応）
# - ファイルアップロード → 即RAG反映（DB保存）
# - コード生成専用エンドポイント (/code)
# - 再起動後もDBからRAG文脈を取得
# - 既存 ./uploaded_files への保存は継続
# ======================================================================

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import requests
import time
import shutil
import ast
import json
import sqlite3
from typing import List, Tuple

# PDF読み込み用
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# ------------------------------------------------------
# アプリ初期化
# ------------------------------------------------------
app = FastAPI(title="RAG WebUI Backend", version="1.3-persist")

# フロントエンド(CORS許可: ローカル開発用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番は適切に制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# 環境設定
# ------------------------------------------------------
OLLAMA_API = "http://localhost:11434/api/generate"
UPLOAD_DIR = "./uploaded_files"
DB_DIR = "./rag_store"
DB_PATH = os.path.join(DB_DIR, "rag_store.sqlite3")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# 1ドキュメントからプロンプトに入れる最大文字数（安全枠）
MAX_SNIPPET_CHARS = 2000
# /ask で結合する最大件数
MAX_CONTEXT_DOCS = 3

# ------------------------------------------------------
# SQLite 永続化レイヤ
# ------------------------------------------------------
def _connect():
    # 毎回短命接続（スレッド安全・自動クローズ）
    return sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)

def init_storage():
    with _connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            path TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_created ON docs(created_at DESC);")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_path ON docs(path);")

def save_document(filename: str, path: str, content: str):
    # 既存ファイルはUPSERT（pathをユニークキー扱い）
    now = time.time()
    with _connect() as conn:
        try:
            conn.execute(
                "INSERT INTO docs(filename, path, content, created_at) VALUES(?,?,?,?)",
                (filename, path, content, now),
            )
        except sqlite3.IntegrityError:
            # 既に登録済み → 更新として扱う
            conn.execute(
                "UPDATE docs SET filename=?, content=?, created_at=? WHERE path=?",
                (filename, content, now, path),
            )

def get_recent_docs(limit: int = MAX_CONTEXT_DOCS) -> List[Tuple[str, str]]:
    # [(filename, content), ...] を新しい順で返す
    with _connect() as conn:
        cur = conn.execute(
            "SELECT filename, content FROM docs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()

def count_docs() -> int:
    with _connect() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM docs")
        return int(cur.fetchone()[0])

# 起動時にDB初期化
init_storage()

# ------------------------------------------------------
# Ollama API 呼び出しヘルパー
# ------------------------------------------------------
def call_ollama_generate(model: str, prompt: str, stream: bool = False):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.0
        }
    }
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=600)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            lines = r.text.strip().splitlines()
            result = ""
            for line in lines:
                try:
                    obj = json.loads(line)
                    result += obj.get("response", "")
                except:
                    continue
            return {"response": result}
    except Exception as e:
        tip = "Ollama サーバが起動しているか確認してください (ollama serve)。"
        raise RuntimeError(f"Ollama API 呼び出し失敗: {e}. {tip}") from e

# ------------------------------------------------------
# RAG検索（DB永続化版）
# ------------------------------------------------------
def retrieve_context(question: str) -> str:
    """
    最新ドキュメントから最大MAX_CONTEXT_DOCS件を連結。
    """
    docs = get_recent_docs(limit=MAX_CONTEXT_DOCS)
    if not docs:
        return f"関連情報なし: {question}"
    chunks = []
    for filename, content in docs:
        snippet = (content or "")[:MAX_SNIPPET_CHARS]
        chunks.append(f"【{filename}】\n{snippet}")
    return "\n\n".join(chunks)

# ------------------------------------------------------
# コード検証ユーティリティ
# ------------------------------------------------------
def validate_python_code(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception as e:
        print(f"[WARN] 構文エラー: {e}")
        return False

# ------------------------------------------------------
# /ask エンドポイント
# ------------------------------------------------------
@app.post("/ask")
async def ask(
    question: str = Form(...),
    model: str = Form("llama3.1:8b"),
):
    try:
        start = time.time()
        context = retrieve_context(question)
        prompt = f"""
以下はユーザからの質問です。アップロード済み資料も参考にして回答してください。

質問:
{question}

参考情報:
{context}

注意:
- 日本語で簡潔に答えること。
"""
        res = call_ollama_generate(model, prompt, stream=False)
        elapsed = round(time.time() - start, 2)
        return {
            "answer": res.get("response", ""),
            "mode": "general",
            "llm": "ollama",
            "llm_model": model,
            "sources": [context],
            "elapsed": elapsed,
        }
    except Exception as e:
        print("[ERROR] /ask failed:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------------------------------------------
# /code エンドポイント
# ------------------------------------------------------
@app.post("/code")
async def generate_code(
    question: str = Form(...),
    model: str = Form("llama3.1:8b"),
):
    try:
        start = time.time()
        prompt = f"""
あなたは熟練したソフトウェアエンジニアです。
次のリクエストに対して、**実行可能なPythonコードのみ**を返してください。
出力は必ず `def ...:` などから始めてください。

厳守事項:
- 出力はPythonコードのみ。
- 説明文やマークダウン( ``` )は不要。
- コメントは最小限OK。
- math.pi など定数は禁止。
- ゼロ除算やインデックスエラーを避けること。

リクエスト:
{question}
"""
        code = None
        for i in range(3):
            res = call_ollama_generate(model, prompt, stream=False)
            candidate = res.get("response", "").strip()
            if candidate.startswith("```"):
                candidate = candidate.strip("`")
                if candidate.startswith("python"):
                    candidate = candidate[len("python"):].strip()
            if validate_python_code(candidate):
                code = candidate
                break
            else:
                print(f"[WARN] 構文検証失敗 (try {i+1}/3)")
        if not code:
            return JSONResponse(status_code=500, content={"error": "構文エラーが解消できませんでした"})
        elapsed = round(time.time() - start, 2)
        return {
            "answer": code,
            "mode": "code",
            "llm": "ollama",
            "llm_model": model,
            "sources": [],
            "elapsed": elapsed,
        }
    except Exception as e:
        print("[ERROR] /code failed:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------------------------------------------
# /upload エンドポイント（★永続化対応）
# ------------------------------------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    ファイルアップロード
    - PDF, TXT を保存（ディスク）
    - 内容を抽出して DB に保存（永続化）
    - /ask 時はDBから参照 → 再起動に耐える
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # 一旦テンポラリに書いてからアトミックリネーム（簡易）
        tmp_path = file_path + ".tmp"
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            buffer.flush()
            os.fsync(buffer.fileno())
        os.replace(tmp_path, file_path)  # アトミック

        # 内容抽出
        content = ""
        if file.filename.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        elif file.filename.lower().endswith(".pdf") and PyPDF2:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    content += page.extract_text() or ""
        else:
            content = f"(非対応ファイル形式: {file.filename})"

        # DBへ保存（永続化）
        save_document(file.filename, os.path.abspath(file_path), content or "")

        return {"filename": file.filename, "status": "uploaded_and_indexed"}
    except Exception as e:
        print("[ERROR] /upload failed:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------------------------------------------
# /reindex エンドポイント（任意）
# 既存 ./uploaded_files を走査してDB未登録のものを取り込む
# ------------------------------------------------------
@app.post("/reindex")
async def reindex():
    try:
        added = 0
        for name in os.listdir(UPLOAD_DIR):
            path = os.path.join(UPLOAD_DIR, name)
            if not os.path.isfile(path):
                continue
            # 既に入っていればUPDATE扱いになる（save_document内のUPSERTに任せる）
            content = ""
            if name.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif name.lower().endswith(".pdf") and PyPDF2:
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        content += page.extract_text() or ""
            else:
                content = f"(非対応ファイル形式: {name})"
            save_document(name, os.path.abspath(path), content or "")
            added += 1
        return {"status": "ok", "reindexed": added, "total_in_db": count_docs()}
    except Exception as e:
        print("[ERROR] /reindex failed:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------------------------------------------
# /models エンドポイント
# ------------------------------------------------------
@app.get("/models")
async def list_models():
    models = [
        {"id": "mistral:7b-instruct", "note": "和文も無難、応答安定。7B級。"},
        {"id": "qwen2.5:3b-instruct", "note": "軽量で速い、使い勝手◎（3B）。"},
        {"id": "phi3:mini", "note": "超軽量。返答の質は文脈次第。"},
        {"id": "gemma:2b-instruct", "note": "軽い。英語寄り、和文要工夫。"},
        {"id": "tinyllama:chat", "note": "超軽量テスト用途向け。"},
        {"id": "neural-chat:7b", "note": "チャット指向で扱いやすい。"},
        {"id": "qwen2:7b-instruct", "note": "Qwen最新系、7B帯の有力。"},
        {"id": "llama2:7b-chat", "note": "枯れて安定。和文は工夫要。"},
        {"id": "llama3.1:8b", "note": "最新 LLaMA3.1 8B、コード生成も得意。"},
    ]
    return {"models": models}

# ------------------------------------------------------
# おまけ: ヘルスチェック
# ------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "docs_in_db": count_docs(),
        "upload_dir": os.path.abspath(UPLOAD_DIR),
        "db_path": os.path.abspath(DB_PATH),
        "pdf_support": bool(PyPDF2),
        "version": "1.3-persist"
    }
