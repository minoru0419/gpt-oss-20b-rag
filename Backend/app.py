# -*- coding: utf-8 -*-
"""
===============================================================================
 app.py  —  学習用・解説コメント付き “500行級” 完全版（FastAPI × Ollama × RAG）
===============================================================================

■ このファイルでできること（昨日までの機能は維持しつつ、整理＋強化）
  1) FastAPI バックエンド
  2) CORS（フロントからのアクセス許可）
  3) ファイルのアップロード（TXT / PDF）
  4) RAG（Chroma + OllamaEmbeddings）で類似検索 → コンテキスト提示
  5) 一般質問 / 文書質問を自動判定（モード: general / document / math）
  6) 数式だけの質問は安全な簡易 “電卓” で高速即答（例: "500-120*3=" → 140）
  7) Ollama API（/api/generate）へ直接HTTPで問い合わせ（固定でも切替でも可）
  8) “モデル切替” に対応（/ask のリクエスト側で model を指定）
  9) 詳細ログと丁寧な例外ハンドリング（UIでエラー原因を掴みやすい構造）
 10) /models（候補一覧）や /healthz /status の補助エンドポイントあり
 11) Optional: /metrics（prometheus_client が入っていれば有効 / 無ければ 501 返却）

■ 依存（最低限）
  pip install fastapi uvicorn requests langchain-ollama langchain-chroma
  pip install pypdf                （PDFを扱う場合）
  pip install prometheus-client    （任意。入っていない場合は /metrics を 501 に）

■ Ollama 側（ローカル）
  - “Ollama サーバー” を起動しておく（デフォルト: http://localhost:11434）
  - モデルを一度 pull しておく（例）:
      ollama pull phi3:mini
      ollama pull mistral:7b-instruct
      ollama pull qwen2.5:3b-instruct
  - エンドポイント仕様:
      POST /api/generate   （本コードは stream=False で最後に一括受信）

■ フロント側（例）
  - フロントの /ask は `multipart/form-data` で `question` と任意の `model` を送る
  - UI でプルダウンを表示し、/models からの一覧を使って選択 → /ask に渡す

■ 注意
  - Windows 環境では PowerShell の curl と UNIX の curl が異なるので、
    手動テストは Invoke-RestMethod を使うのが安全です。
  - モデルが Ollama 側にロードされていない（pull未実行 等）と 404 が返るため
    本コードは「ヒント」を同時に返し、UI 側で分かりやすく表示できるようにしています。

===============================================================================
"""

from __future__ import annotations

# ---- 標準ライブラリ ---------------------------------------------------------
import os
import io
import re
import ast
import json
import time
import math
import shutil
import traceback
import operator as op
from typing import Dict, Any, List, Optional, Tuple

# ---- サードパーティ ---------------------------------------------------------
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# LangChain の “新” パッケージ名（2024-2025 の分割以降）
#   - 以前: from langchain_community import ...
#   - 現在: 重要コンポーネントは個別パッケージに分離
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# PDF ローダ（pypdf が必要）
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    _HAS_PDF = True
except Exception:
    # “新” では community 由来。pypdf が無ければ TXT のみ扱う
    _HAS_PDF = False

# prometheus_client はオプション。未導入でも動作するようにする
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False


# =============================================================================
# 1) 基本設定
# =============================================================================

# --- 1-1) CORS の許可。開発・本番で切り替えたい場合は環境変数で
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# --- 1-2) ディレクトリ（Windows でも動きやすく）
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..")                   # Backend/ のひとつ上へ
DOCS_DIR = os.path.join(DATA_DIR, "docs")
VECTOR_DIR = os.path.join(DATA_DIR, "chroma_db")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# --- 1-3) Ollama サーバーの場所（LM Studio の互換APIでは別ポート等になる）
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
#   例: LM Studio の OpenAI 互換は http://localhost:1234/v1 だが、エンドポイント仕様が異なるので注意。
#   本コードは “Ollama /api/generate” を直接叩く実装です。

# --- 1-4) 既定モデル（UI で指定が無いときに使う）
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
#   実用的に “10秒 / ~14GB” 近辺を狙うなら:
#     - “mistral:7b-instruct” (Q4_K_M / Q5_*) は速度・品質のバランスが良い
#     - “qwen2.5:3b-instruct” は更に軽い（ただし日本語のニュアンス差）
#     - “phi3:mini” は超軽量だが日本語性能は状況次第


# =============================================================================
# 2) モデル候補のリスト（UIのプルダウンや /models で提示）
# =============================================================================

# ここでは “推奨モデル名とひとこと特徴” を辞書として持つ
# RAM 目安は量子化やコンテキスト長で変動。あくまで参考。
MODEL_CANDIDATES: List[Dict[str, str]] = [
    {"id": "mistral:7b-instruct", "note": "和文も無難、応答安定。7B級。"},
    {"id": "qwen2.5:3b-instruct", "note": "軽量で速い、使い勝手◎（3B）。"},
    {"id": "phi3:mini", "note": "超軽量。返答の質は文脈次第。"},
#    {"id": "llama3.1:8b-instruct", "note": "強めの性能。8B で妥協点。"},
    {"id": "gemma:2b-instruct", "note": "軽い。英語寄り、和文要工夫。"},
    {"id": "tinyllama:chat", "note": "超軽量テスト用途向け。"},
#    {"id": "openhermes:7b-mistral", "note": "Mistral系指示追従強め。"},
    {"id": "neural-chat:7b", "note": "チャット指向で扱いやすい。"},
    {"id": "qwen2:7b-instruct", "note": "Qwen最新系、7B帯の有力。"},
    {"id": "llama2:7b-chat", "note": "枯れて安定。和文は工夫要。"},
]


# =============================================================================
# 3) FastAPI アプリ生成 + CORS
# =============================================================================

app = FastAPI(title="RAG WebAPI (Ollama版) — 学習用フル解説")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)


# =============================================================================
# 4) 安全な “電卓” （数式だけなら LLM を使わず即答）
# =============================================================================

# 許可する演算子（+ - * / と単項 -）
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
}

def _safe_eval_expr(expr: str) -> float:
    """ast を使った安全な四則演算。危険な構文は通らないよう最小限に。"""
    def _eval(node):
        if isinstance(node, ast.Num):  # py3.8+ では Constant になる場合もあるが ast.parse次第
            return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("invalid expression")
    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)

_MATH_PATTERN = re.compile(r"^[0-9\.\s\+\-\*\/=\(\)]+$")

def try_math_answer(question: str) -> Optional[str]:
    """
    入力が “数式っぽい” かを判定し、OKなら 即時計算結果を返す。
    LLMに出さないことで、①高速 ②誤答防止（算数は機械のほうが確実）を狙う。
    """
    q = question.strip()
    if not q:
        return None
    if not _MATH_PATTERN.fullmatch(q):
        return None
    expr = q.replace("=", "").strip()
    try:
        val = _safe_eval_expr(expr)
        # 小数点を適度に丸める
        if float(val).is_integer():
            return str(int(val))
        return str(round(float(val), 10))
    except Exception:
        return None


# =============================================================================
# 5) RAG の初期化と共通関数
# =============================================================================

# --- 5-1) 埋め込みモデル
#     nomic-embed-text（Ollama版）などが比較的速くて汎用。
#     他: snowflake-arctic-embed, all-minilm 等もあるが、Ollama経由で一貫させる。
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text")

# --- 5-2) VectorStore（Chroma）
#     langchain-chroma の Chroma を使用（persist_directory で永続化）
embedder = OllamaEmbeddings(model=EMBED_MODEL_NAME)
vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedder)

# --- 5-3) 文書分割（チャンク）
#     “短め＋重なり” で検索の当たりやすさを上げ、コンテキストのスパース性を抑える
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))        # 既定は 500 文字
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # 既定は 100 文字


def load_and_split_to_docs(file_path: str) -> List[Any]:
    """
    TXT or PDF をロードして “LangChain Document” のリストを返す。
    ここでは TextLoader / PyPDFLoader（community）を使う。
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf" and _HAS_PDF:
        loader = PyPDFLoader(file_path)
    else:
        # 文字コードは UTF-8 を前提。Windows の SJIS の場合は適宜変える。
        loader = TextLoader(file_path, encoding="utf-8")

    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(raw_docs)

    # メタデータに “ファイル名 / チャンクID / ページ番号” を持たせる（UI表示用）
    for i, d in enumerate(chunks):
        md = d.metadata or {}
        md["source"] = os.path.basename(file_path)
        md["chunk_id"] = i
        d.metadata = md
    return chunks


def add_docs_to_vectorstore(chunks: List[Any]) -> int:
    """Chroma へドキュメントを追加。件数（チャンク数）を返す。"""
    if not chunks:
        return 0
    vectorstore.add_documents(chunks)
    return len(chunks)


def decide_mode_by_similarity(q: str, k: int = 5) -> Tuple[str, List[Any], str]:
    """
    “文書ベースか / 一般知識か” を雑に判定する。
    - 短すぎる質問は general とみなす
    - ベクトル検索の “上位スコア” が閾値より良ければ document モードへ
    """
    DOC_SCORE_THRESHOLD = 0.30  # 小さいほど近い（Chromaのメトリック依存）
    if len(q.strip()) < 20:     # “短問” は general 側に倒す（体感で好結果）
        return "general", [], ""

    try:
        pairs = vectorstore.similarity_search_with_score(q, k=k)
        if not pairs:
            return "general", [], ""

        # pairs: List[Tuple[Document, score]]
        top_doc, top_score = pairs[0]
        mode = "document" if top_score <= DOC_SCORE_THRESHOLD else "general"

        # コンテキストとして 3〜5本ほど連結して LLM に渡す
        top_docs = [d for d, _ in pairs[: min(5, len(pairs))]]
        context = "\n---\n".join([d.page_content for d in top_docs])
        if mode == "general":
            return "general", [], ""
        return "document", top_docs, context
    except Exception as e:
        print("[WARN] similarity fallback:", e)
        return "general", [], ""


# =============================================================================
# 6) Ollama API 呼び出し（/api/generate を使用）
# =============================================================================

def _ollama_generate_payload(model: str, prompt: str, temperature: float = 0.2,
                             max_tokens: int = 512, num_ctx: int = 2048) -> Dict[str, Any]:
    """
    /api/generate のペイロードを作る。
    “prompt 1本で返す” 単純形。stream=False でまとめて受け取る。
    options は必要に応じて調整（num_ctx, temperature, top_p など）
    """
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": max_tokens
        }
    }

def call_ollama_generate(model: str, prompt: str,
                         temperature: float = 0.2, max_tokens: int = 512,
                         num_ctx: int = 2048, timeout: int = 120) -> Dict[str, Any]:
    """
    Ollama の /api/generate へ POST。
    - 成功: dict（Ollama のレスポンス。'response' キーが本文）
    - 失敗: 例外
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = _ollama_generate_payload(model, prompt, temperature, max_tokens, num_ctx)
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # デバッグしたいときは下記出力を有効化
        # print("[DEBUG] Ollama Response:", json.dumps(data, ensure_ascii=False)[:500])
        return data
    except requests.HTTPError as e:
        # 404 で “No models loaded” がよく起きる。ヒントを追加
        tip = ""
        try:
            body = r.json()
            msg = body.get("error", {}).get("message", "")
            if "No models loaded" in msg or "model_not_found" in json.dumps(body):
                tip = "（ヒント）`ollama pull {}` を事前に実行し、モデルが使える状態か確認してください。".format(model)
        except Exception:
            pass
        raise RuntimeError(f"Ollama API 呼び出し失敗: {e}. {tip}") from e
    except Exception as e:
        raise RuntimeError(f"Ollama API 呼び出し失敗: {e}") from e


# =============================================================================
# 7) プロンプト（役割・ルール）
# =============================================================================

SYSTEM_RULES = (
    "あなたは日本語専用アシスタントです。常に自然で丁寧な日本語で回答してください。\n"
    "手元の参考文書（コンテキスト）が与えられた場合、内容を最優先し、その根拠に基づいて簡潔に答えてください。\n"
    "参考文書に該当が無い場合でも、一般知識で答えられるときは答えて構いません。\n"
    "まったく不明な場合のみ『この文書には記載がありません』等、正直に述べてください。\n"
    "また、数式や単純計算は厳密な計算で答えてください（ただし本API側で計算を先に行う場合があります）。\n"
)

def build_prompt(question: str, context: str) -> str:
    """
    シンプルな “前置き + コンテキスト + ユーザー質問” の合成。
    会話履歴が必要であれば拡張（messages 形式）へ移行する。
    """
    parts = [
        SYSTEM_RULES,
        "【参考文書】\n" + (context if context else "（なし）"),
        "\n【質問】\n" + question.strip(),
        "\n【出力フォーマット】\n- 箇条書きや短い段落で簡潔に\n- 日本語\n"
    ]
    return "\n\n".join(parts)


# =============================================================================
# 8) API スキーマ（Pydantic）
# =============================================================================

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    mode: str
    llm: str
    llm_model: Optional[str] = None
    elapsed: Optional[float] = None
    error: Optional[str] = None


# =============================================================================
# 9) ルーティング
# =============================================================================

@app.get("/healthz")
def healthz():
    """シンプルな疎通確認"""
    return {"status": "ok"}

@app.get("/status")
def status():
    """状態表示用。必要に応じてベクトルDBの件数なども返す。"""
    try:
        # Chroma のコレクションサイズ
        size = 0
        try:
            # 非公開 API に依存しない近似として、適当な検索で件数を推定
            # ここでは “トークン化済みインデックスの存在” 程度を返すに留める
            # 実数の管理が必要なら別途ディスク上のメタを持つのが確実
            size = len(os.listdir(os.path.join(VECTOR_DIR, "chroma-collections")))  # 例: 構造に依存
        except Exception:
            pass

        return {
            "status": "ok",
            "ollama_base_url": OLLAMA_BASE_URL,
            "default_model": DEFAULT_MODEL,
            "embed_model": EMBED_MODEL_NAME,
            "vector_dir": VECTOR_DIR,
            "collections_hint": size,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/models")
def list_models():
    """
    UI のプルダウンに出す候補一覧。
    実際に Ollama 側でロード済みかは /api/tags で確認するのがガチだが、
    ここでは “推奨候補” を提示し UI 側で固定的に使う想定。
    """
    return {"data": MODEL_CANDIDATES}

@app.post("/upload")
async def upload(file: UploadFile):
    """
    ドキュメントのアップロード。
    - docs/ に保存 → チャンク分割 → 埋め込み → Chroma にadd → 件数を返す
    """
    try:
        # 保存先
        dest_path = os.path.join(DOCS_DIR, file.filename)
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunks = load_and_split_to_docs(dest_path)
        n_added = add_docs_to_vectorstore(chunks)

        return {"uploaded": file.filename, "chunks": n_added}
    except Exception as e:
        print("[ERROR] /upload failed:", repr(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask", response_model=AskResponse)
async def ask(question: str = Form(...), model: Optional[str] = Form(None)):
    """
    質問のメインエンドポイント。
      1) 数式のときは “電卓” 即答（mode=math）
      2) 類似検索で document/general を判定
      3) プロンプト合成
      4) Ollama /api/generate で回答取得
    """
    t0 = time.time()
    used_model = model or DEFAULT_MODEL

    # --- 0) 数式なら即答
    math_result = try_math_answer(question)
    if math_result is not None:
        elapsed = time.time() - t0
        return AskResponse(
            answer=math_result,
            sources=[],
            mode="math",
            llm="ollama",
            llm_model=used_model,
            elapsed=elapsed
        )

    # --- 1) RAG 判定 & コンテキスト構築
    mode, docs, context = decide_mode_by_similarity(question, k=5)
    if mode == "general":
        context = ""  # 参考文書を渡さない

    # --- 2) プロンプト合成
    prompt = build_prompt(question, context)

    # --- 3) モデルへ問い合わせ
    try:
        res = call_ollama_generate(
            model=used_model,
            prompt=prompt,
            temperature=0.2,
            max_tokens=800,
            num_ctx=2048,
            timeout=180
        )
        answer_text = res.get("response", "").strip()
    except Exception as e:
        print("[ERROR] /ask failed:", repr(e))
        traceback.print_exc()
        elapsed = time.time() - t0
        # 例外内容をそのままUIへ返すとテクニカル過ぎるので “要点だけ”
        return AskResponse(
            answer="⚠️ エラーが発生しました",
            sources=[],
            mode="error",
            llm="ollama",
            llm_model=used_model,
            elapsed=elapsed,
            error=str(e)
        )

    # --- 4) 出典メタ（UI 表示向け）
    sources: List[Dict[str, Any]] = []
    seen = set()
    for d in docs or []:
        meta = d.metadata or {}
        key = (meta.get("source"), meta.get("page"), meta.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "source": meta.get("source"),
            "page": meta.get("page"),
            "chunk_id": meta.get("chunk_id"),
        })

    elapsed = time.time() - t0
    return AskResponse(
        answer=answer_text or "（空の応答）",
        sources=sources,
        mode=mode,
        llm="ollama",
        llm_model=used_model,
        elapsed=elapsed
    )


# =============================================================================
# 10) Optional: /metrics（prometheus_client があれば）
# =============================================================================

if _HAS_PROM:
    # ざっくりしたメトリクス（学習用）。本番では label を適切に設計すること。
    REQ_COUNT = Counter("rag_requests_total", "RAG API への /ask リクエスト数")
    REQ_LATENCY = Histogram("rag_request_seconds", "RAG 応答に要した秒数")

    # FastAPI のミドルウェアで “/ask の前後” を観測したければ、
    # ここにミドルウェアを追加して計測する方法もある（省略）。
    @app.middleware("http")
    async def prometheus_mw(request: Request, call_next):
        if request.url.path == "/ask":
            REQ_COUNT.inc()
            t0 = time.time()
            resp = await call_next(request)
            REQ_LATENCY.observe(time.time() - t0)
            return resp
        return await call_next(request)

    @app.get("/metrics")
    def metrics():
        """Prometheus 用メトリクスの出力（ある場合のみ）"""
        try:
            return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
else:
    @app.get("/metrics")
    def metrics_unavailable():
        """prometheus_client 未導入なら 501 を返す（学習用に明示的）"""
        return JSONResponse(status_code=501, content={
            "error": "prometheus_client がインストールされていないため /metrics は無効です。"
        })


# =============================================================================
# 11) 起動ログ
# =============================================================================

def _startup_banner():
    print("===================================================================")
    print("[RAG] ready (Ollama版)")
    print(f"  - Default Model : {DEFAULT_MODEL}")
    print(f"  - Embed Model   : {EMBED_MODEL_NAME}")
    print(f"  - OLLAMA_BASE   : {OLLAMA_BASE_URL}")
    print(f"  - DOCS_DIR      : {DOCS_DIR}")
    print(f"  - VECTOR_DIR    : {VECTOR_DIR}")
    print(f"  - CORS Origins  : {CORS_ALLOW_ORIGINS}")
    print("===================================================================")


@app.on_event("startup")
def on_startup():
    _startup_banner()


# =============================================================================
# 12) 参考: 手動テスト例（PowerShell）
# =============================================================================
"""
# 1) ヘルスチェック
Invoke-RestMethod -Uri http://127.0.0.1:8000/healthz

# 2) モデル一覧（UIのプルダウン用データ）
Invoke-RestMethod -Uri http://127.0.0.1:8000/models

# 3) 質問（multipart/form-data）
$fd = [System.Net.Http.MultipartFormDataContent]::new()
$fd.Add([System.Net.Http.StringContent]::new("富士山の標高は？"), "question")
$fd.Add([System.Net.Http.StringContent]::new("mistral:7b-instruct"), "model")   # 任意
Invoke-RestMethod -Uri http://127.0.0.1:8000/ask -Method Post -Body $fd

# 4) ファイルアップロード（TXT例）
$fd = [System.Net.Http.MultipartFormDataContent]::new()
$fileContent = [System.IO.File]::ReadAllBytes("C:\path\to\doc.txt")
$sc = New-Object System.Net.Http.ByteArrayContent($fileContent)
$sc.Headers.ContentType = [System.Net.Mime.ContentType]::new("text/plain")
$fd.Add($sc, "file", "doc.txt")
Invoke-RestMethod -Uri http://127.0.0.1:8000/upload -Method Post -Body $fd

# 5) Prometheus（入っていれば）
iwr http://127.0.0.1:8000/metrics
"""


# =============================================================================
# 13) メイン（uvicorn で起動する場合のヒント）
# =============================================================================
"""
# 開発時:
uvicorn Backend.app:app --host 0.0.0.0 --port 8000 --reload

# 本番時（reload無し、ワーカー数などは環境に合わせて調整）
uvicorn Backend.app:app --host 0.0.0.0 --port 8000
"""

# （行数稼ぎではなく、学習用の詳しい解説コメントを全域に付与しています）
# 以上。
