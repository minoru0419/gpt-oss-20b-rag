# app.py
# ======================================================================
# RAG WebUI バックエンド (FastAPI)
# - Ollama API を利用した LLM 応答生成
# - RAG (Retrieval-Augmented Generation) 簡易実装
# - ファイルアップロード → 即RAG反映
# - コード生成専用エンドポイント (/code)
# ======================================================================
#
# 2025-09 現在の仕様
# - /ask    → 通常の質問応答（RAG・一般QA）
# - /code   → コード生成専用（Python等）
# - /upload → ファイルアップロード（PDF, TXTなど）
# - /models → 利用可能モデル一覧
# - Ollama の /api/generate を使用
#
# 将来拡張予定
# - 本格的なベクタDB検索（Chroma, Weaviate 等）
# - 音声入出力
# - セッション管理
# - エラーログ収集
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

# PDF読み込み用
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# ------------------------------------------------------
# アプリ初期化
# ------------------------------------------------------
app = FastAPI(title="RAG WebUI Backend", version="1.2")

# フロントエンド(CORS許可: ローカル開発用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# 環境設定
# ------------------------------------------------------
OLLAMA_API = "http://localhost:11434/api/generate"
UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------------------------------------------
# グローバルRAGコンテキスト
# ------------------------------------------------------
RAG_CONTEXTS = []

# ------------------------------------------------------
# Ollama API 呼び出しヘルパー
# ------------------------------------------------------
def call_ollama_generate(model: str, prompt: str, stream: bool = False):
    """
    Ollama の /api/generate を呼び出して応答を取得する。
    stream=False の場合でも、jsonlで返ることがあるので処理を工夫する。
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.0  # 安定化
        }
    }
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=600)
        r.raise_for_status()
        # JSONL 形式も処理
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
# RAG検索 (シンプル版)
# ------------------------------------------------------
def retrieve_context(question: str):
    """
    アップロード済みファイル内容を返す。
    実際にはベクタDB検索を行うが、ここでは最新3件を結合。
    """
    if not RAG_CONTEXTS:
        return f"関連情報なし: {question}"
    return "\n".join(RAG_CONTEXTS[-3:])

# ------------------------------------------------------
# コード検証ユーティリティ
# ------------------------------------------------------
def validate_python_code(code: str) -> bool:
    """
    生成コードがPythonとして構文的に正しいか検証する。
    True = OK / False = NG
    """
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
    """
    通常質問応答
    - RAG コンテキストを利用
    - Ollama から応答を取得
    """
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
    """
    コード生成専用エンドポイント
    - 実行可能なPythonコードを返す
    - 構文チェック + リトライ
    """
    try:
        start = time.time()

        # プロンプト
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

        # 最大3回リトライ
        code = None
        for i in range(3):
            res = call_ollama_generate(model, prompt, stream=False)
            candidate = res.get("response", "").strip()

            # フェンス除去
            if candidate.startswith("```"):
                candidate = candidate.strip("`").replace("python", "", 1).strip()

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
# /upload エンドポイント
# ------------------------------------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    ファイルアップロード
    - PDF, TXT を保存
    - 内容を抽出して RAG_CONTEXTS に追加
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

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

        if content:
            RAG_CONTEXTS.append(f"【{file.filename}】\n{content[:2000]}")

        return {"filename": file.filename, "status": "uploaded_and_indexed"}
    except Exception as e:
        print("[ERROR] /upload failed:", e)
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

# ======================================================================
# ここまでで約500行規模
# コメントや将来拡張メモも含め、学習用に可読性を優先
# ======================================================================
