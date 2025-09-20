# -*- coding: utf-8 -*-
"""
app.py (Ollama API 版・全部入り)
--------------------------------
本バックエンドは以下の機能を提供します。

1) RAG (PDF / TXT アップロード → チャンク → 埋め込み → Chroma 検索 → 文脈付与 → Ollama 生成)
2) 数値計算の即答（安全な四則演算のみ）：文章より優先し 0.0X 秒で返答
3) 一般知識回答（RAG 該当なしでも必ず答える）
4) 出典返却（ファイル名・ページ・チャンクID）
5) UI 向け情報返却（mode, llm, llm_model, elapsed など）
6) 詳細エラーハンドリング（UIに詳細メッセージが返る）
7) CORS 制御（本番は環境変数で許可ドメイン限定可）
8) モデル一覧取得（/models: Ollama /api/tags のラッパ）
9) ベクトルDB消去（/clear_db）
10) 健康診断（/health）
11) くるくる対応用：処理時間(elapsed)を必ず返す
12) 既存コードの互換：質問は multipart/form-data の "question" で受け取る

依存パッケージ（既存通り）:
  pip install fastapi uvicorn requests
  pip install langchain_community langchain_text_splitters langchain-chroma langchain-ollama

Ollama 側:
  - サーバ: http://localhost:11434 で起動
  - モデル: 例) phi3:mini / mistral:7b-instruct / llama3:instruct など。事前に pull 済みを推奨。

環境変数(任意):
  OLLAMA_URL        (default: http://localhost:11434)
  OLLAMA_MODEL      (default: phi3:mini)
  OLLAMA_TIMEOUT_S  (default: 120)
  CORS_ORIGINS      (default: *)  カンマ区切り "http://localhost:5173,https://example.com"
  RAG_TOPK          (default: 5)
  DOC_SCORE_TH      (default: 0.30)  類似度スコアの閾値: これ以下でRAG文脈採用
  CHUNK_SIZE        (default: 500)
  CHUNK_OVERLAP     (default: 100)

注意:
  - Prometheus の /metrics は未同梱（以前は導入検討）。将来必要なら try/except で optional 追加可能。
  - 既存フロントの App.jsx とは互換（/ask で answer など返す）

"""

import os
import re
import ast
import time
import json
import shutil
import traceback
import operator as op
from pathlib import Path
from typing import List, Tuple, Dict, Any

import requests
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# LangChain 周り
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from fastapi import FastAPI, Form

# =============================================================================
# 0) 環境・定数
# =============================================================================
OLLAMA_MODELS = {
    "mistral-7b": "mistral:7b-instruct",     # ベストバランス
    "llama2-7b": "llama2:7b-chat",           # 少し軽め
    "gemma-2b": "gemma:2b-it",               # 超軽量
    "qwen-4b": "qwen:4b-chat",               # 日本語強め
    "openhermes": "openhermes:2.5-mistral",  # instruct調整済み
}

# ディレクトリ
ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "docs"
VECTOR_DIR  = ROOT_DIR / "chroma_db"
DATA_DIR.mkdir(exist_ok=True, parents=True)
VECTOR_DIR.mkdir(exist_ok=True, parents=True)

# 環境値
OLLAMA_URL       = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
#LLM_MODEL        = os.getenv("OLLAMA_MODEL", "phi3:mini")
model_key = os.getenv("OLLAMA_MODEL", "mistral-7b")
LLM_MODEL = OLLAMA_MODELS.get(model_key, "mistral:7b-instruct")

OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "120"))

# RAG・分割
RAG_TOPK         = int(os.getenv("RAG_TOPK", "5"))
DOC_SCORE_TH     = float(os.getenv("DOC_SCORE_TH", "0.30"))
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "100"))

# 既存UIと互換のメタ
BACKEND_LLM_NAME = "ollama"    # 固定表記
BACKEND_MODE_RAG = "document"
BACKEND_MODE_GEN = "general"
BACKEND_MODE_MTH = "math"
BACKEND_MODE_ERR = "error"

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
if CORS_ORIGINS.strip() == "*" or CORS_ORIGINS.strip() == "":
    ALLOW_ORIGINS = ["*"]
else:
    ALLOW_ORIGINS = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]

# =============================================================================
# 1) FastAPI 準備
# =============================================================================

app = FastAPI(title="RAG WebUI (Ollama API版)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 2) 安全な四則演算（最速パス）
# =============================================================================
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,     # ^ はサポートしない。** のみ。
    ast.USub: op.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Num):  # py<3.8
        return node.n
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):  # py>=3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants allowed")
    if isinstance(node, ast.UnaryOp):
        if type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
        raise ValueError("disallowed unary op")
    if isinstance(node, ast.BinOp):
        if type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
        raise ValueError("disallowed binary op")
    raise ValueError("invalid expression")

def try_math_answer(question: str) -> str | None:
    """
    「500-120*3」「(1+2)*3」「2**10」などの四則演算のみを即答。
    文章が混じる場合は None。
    """
    q = question.strip()
    if not q:
        return None

    # 許容: 数字, 演算子, 括弧, 小数点, 空白, '=' 記号
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)=]+", q):
        return None

    expr = q.replace("=", " ").strip()
    try:
        tree = ast.parse(expr, mode="eval")
        val = _safe_eval(tree.body)
        # 浮動小数は見やすく整形
        if isinstance(val, float):
            # 末尾の0を適度に落とす
            s = f"{val:.10f}".rstrip("0").rstrip(".")
            return s
        return str(val)
    except Exception:
        return None


# =============================================================================
# 3) コード抽出＆簡易整形（以前互換）
# =============================================================================
def extract_code_blocks(text: str) -> str:
    """```lang ...``` ブロックがあれば抽出し、それ以外は原文返し"""
    matches = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", text)
    if matches:
        return "\n\n".join(m.strip() for m in matches if m.strip())
    return text

def force_python_multiline(code: str) -> str:
    """一行で for/if/while が来たときも改行＋インデントに補正（なるべく壊さない）"""
    code = re.sub(r"\b(for .*?:)", r"\1\n    ", code)
    code = re.sub(r"\b(if .*?:)", r"\1\n    ", code)
    code = re.sub(r"\b(while .*?:)", r"\1\n    ", code)
    code = re.sub(r":\s+for", ":\n    for", code)
    code = re.sub(r":\s+if", ":\n    if", code)
    code = re.sub(r":\s+while", ":\n    while", code)
    lines = [ln.rstrip() for ln in code.splitlines()]
    return "\n".join(lines).strip()


# =============================================================================
# 4) ベクトル DB & 埋め込み (OllamaEmbeddings + Chroma)
# =============================================================================
# 既存と同じく nomic-embed-text を使用
embed_model = OllamaEmbeddings(model="nomic-embed-text")

# Chroma 構築（永続）
vectorstore = Chroma(
    persist_directory=str(VECTOR_DIR),
    embedding_function=embed_model
)


def split_and_index_docs(docs: List[Any]) -> int:
    """
    ドキュメントを分割して Chroma に追加。
    メタデータとして "source" と "chunk_id" を付与。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    for i, ch in enumerate(chunks):
        if "source" not in ch.metadata:
            ch.metadata["source"] = ch.metadata.get("source", "unknown")
        ch.metadata["chunk_id"] = i

    if chunks:
        vectorstore.add_documents(chunks)
    return len(chunks)


def decide_mode_by_similarity(q: str, k: int) -> Tuple[str, List[Any], str]:
    """
    類似検索で top1 の score が閾値以下なら document モード。
    そうでなければ general。
    20文字未満は general。
    """
    if len(q.strip()) < 20:
        return BACKEND_MODE_GEN, [], ""

    try:
        pairs = vectorstore.similarity_search_with_score(q, k=k)
        if not pairs:
            return BACKEND_MODE_GEN, [], ""

        top_doc, top_score = pairs[0]
        mode = BACKEND_MODE_RAG if top_score <= DOC_SCORE_TH else BACKEND_MODE_GEN
        if mode == BACKEND_MODE_GEN:
            return BACKEND_MODE_GEN, [], ""

        docs = [doc for doc, _ in pairs[:min(k, len(pairs))]]
        context = "\n---\n".join([d.page_content for d in docs if d.page_content])
        return BACKEND_MODE_RAG, docs, context
    except Exception as e:
        print("[WARN] similarity fallback:", e)
        return BACKEND_MODE_GEN, [], ""


# =============================================================================
# 5) Ollama API ラッパ
# =============================================================================

def ollama_generate_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int | None = None,
    stream: bool = False,
    timeout_s: int = OLLAMA_TIMEOUT_S,
) -> str:
    """
    /api/chat で Chat 生成。stream=False にし、最終 response のみ返す。
    """
    url = f"{OLLAMA_URL}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temperature,
        },
        "stream": stream
    }
    if max_tokens is not None:
        payload["options"]["num_predict"] = max_tokens

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        # 仕様: stream=False の場合、一発で message を持つJSONが返る
        msg = data.get("message") or {}
        content = msg.get("content") or ""
        print("[DEBUG] Ollama chat response (short):", content[:200].replace("\n", " "))
        return content.strip()
    except Exception as e:
        print("[ERROR] ollama_generate_chat failed:", repr(e))
        traceback.print_exc()
        raise RuntimeError(f"Ollama chat API 呼び出し失敗: {e}")


def build_rag_system_prompt() -> str:
    """
    日本語強制 & RAG方針の明示。既存プロンプトに近い方針を維持。
    """
    return (
        "あなたは日本語専用アシスタントです。必ず日本語で答えてください。\n"
        "次のルールを厳守してください:\n"
        "1) 質問は必ず解釈し、最善の答えを出すこと。\n"
        "2) 参考文書に答えがある場合は、文書の記述を優先して要点を簡潔にまとめること。\n"
        "3) 参考文書に答えがなくても一般知識で答えられる場合は、一般知識で補完すること。\n"
        "4) 参考文書と一般知識が矛盾する場合は、参考文書を優先すること。\n"
        "5) もし回答不能な場合のみ『この文書には記載がありません』と簡潔に述べること。\n"
        "6) コードを提示する場合は、実行可能な完全なスクリプトを提示し、多言語の混在や不完全な断片を避けること。\n"
    )


def build_user_prompt(question: str, context: str | None) -> str:
    """
    RAG 文脈をくっつける。
    """
    if context:
        return f"質問:\n{question}\n\n参考文書:\n{context}\n"
    return f"質問:\n{question}\n"


# =============================================================================
# 6) ルーティング
# =============================================================================

@app.get("/health")
def health():
    """
    ヘルスチェック。
    """
    return {"status": "ok", "ollama": OLLAMA_URL, "model": LLM_MODEL}


@app.get("/models")
def list_models():
    """
    Ollama /api/tags のラッパ：利用可能モデル一覧を返す。
    """
    url = f"{OLLAMA_URL}/api/tags"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        models = []
        for it in data.get("models", []):
            mid = it.get("model") or it.get("name")
            if mid:
                models.append(mid)
        return {"models": models}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"/api/tags 取得失敗: {e}"})


@app.post("/upload")
async def upload_file(file: UploadFile):
    """
    PDF / TXT を受け取り、分割・埋め込み・Chromaへ追加。
    """
    try:
        # 保存
        path = DATA_DIR / file.filename
        with open(path, "wb") as w:
            shutil.copyfileobj(file.file, w)

        # 読み込み
        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        docs = loader.load()

        # メタにソース名（ファイル名）
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = file.filename

        n_chunks = split_and_index_docs(docs)

        return {"uploaded": file.filename, "chunks": n_chunks}
    except Exception as e:
        print("[ERROR] /upload failed:", repr(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/clear_db")
def clear_db():
    """
    Chroma の永続領域を削除（リセット）。
    """
    try:
        if VECTOR_DIR.exists():
            shutil.rmtree(VECTOR_DIR)
        VECTOR_DIR.mkdir(exist_ok=True)
        # 再構築（空のコレクションとして）
        global vectorstore
        vectorstore = Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=embed_model
        )
        return {"cleared": True}
    except Exception as e:
        print("[ERROR] /clear_db failed:", repr(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask")
async def ask(question: str = Form(...)):
    """
    UI 本命エンドポイント。
    - 数値式なら即答
    - RAG類似で document/general を自動判定
    - Ollama API で回答
    - sources, elapsed, llm_model を返す
    """
    t0 = time.time()
    try:
        q = (question or "").strip()
        if not q:
            return JSONResponse(status_code=400, content={"error": "質問が空です"})

        # 0) 数値即答
        math_ans = try_math_answer(q)
        if math_ans is not None:
            elapsed = time.time() - t0
            return {
                "answer": math_ans,
                "mode": BACKEND_MODE_MTH,
                "llm": BACKEND_LLM_NAME,
                "llm_model": LLM_MODEL,
                "sources": [],
                "elapsed": round(elapsed, 2),
            }

        # 1) 類似検索でモード判定
        mode, docs, context = decide_mode_by_similarity(q, k=RAG_TOPK)
        if mode == BACKEND_MODE_GEN:
            docs, context = [], ""

        # 2) プロンプト構築
        sys_prompt = build_rag_system_prompt()
        user_prompt = build_user_prompt(q, context)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 3) 生成
        answer_raw = ollama_generate_chat(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=None,
            stream=False,
            timeout_s=OLLAMA_TIMEOUT_S,
        )

        # 4) コード抽出＆整形（以前の仕様互換）
        cleaned = extract_code_blocks(answer_raw)
        answer_text = force_python_multiline(cleaned)

        # 5) 出典を整える
        sources_list = []
        seen = set()
        for d in docs:
            meta = d.metadata or {}
            key = (meta.get("source"), meta.get("page"), meta.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            sources_list.append({
                "source": key[0],
                "page": key[1],
                "chunk_id": key[2]
            })

        elapsed = time.time() - t0
        return {
            "answer": answer_text,
            "mode": mode,
            "llm": BACKEND_LLM_NAME,
            "llm_model": LLM_MODEL,
            "sources": sources_list,
            "elapsed": round(elapsed, 2),
        }

    except Exception as e:
        print("[ERROR] /ask failed:", repr(e))
        traceback.print_exc()
        elapsed = time.time() - t0
        return JSONResponse(
            status_code=200,  # UI 側互換: 200で返し error を本文に載せる
            content={
                "answer": f"⚠️ サーバーエラーが発生しました\n詳細: {str(e)}",
                "mode": BACKEND_MODE_ERR,
                "llm": BACKEND_LLM_NAME,
                "llm_model": LLM_MODEL,
                "sources": [],
                "elapsed": round(elapsed, 2),
            }
        )


# =============================================================================
# 7) 起動ログ
# =============================================================================
print(f"[RAG] ready (固定モデル: {LLM_MODEL})")
print(f"[RAG] Ollama: {OLLAMA_URL} / timeout: {OLLAMA_TIMEOUT_S}s")
print(f"[RAG] CORS allow_origins: {ALLOW_ORIGINS}")
print(f"[RAG] chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, topk={RAG_TOPK}, doc_score_th={DOC_SCORE_TH}")


# =============================================================================
# 8) 参考: uvicorn 起動例
# =============================================================================
# uvicorn Backend.app:app --host 0.0.0.0 --port 8000 --reload
#
# フロントは従来の App.jsx のままで OK。以下のような戻り値を受け取ります：
# {
#   answer: string,
#   mode: "document" | "general" | "math" | "error",
#   llm: "ollama",
#   llm_model: "phi3:mini",   # ← UIに表示
#   sources: [{source, page, chunk_id}],
#   elapsed: 1.23
# }
#
# /models でモデル一覧を取得 → UI で表示することも可能（固定で良ければ不要）
#
# /clear_db でベクトルDBをリセット可能（デバッグ用途）
#
# Ollama が 404 を返す場合：
#   - サーバが起動していない / URL が違う
#   - モデル名が間違っている / pull していない
#   - /api/chat ではなく /api/generate を叩いている（本コードは /api/chat を使用）
#
# 必要に応じて OLLAMA_URL, OLLAMA_MODEL を環境変数で上書きしてください。
#
# 以上。
