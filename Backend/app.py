# app.py
# ======================================================================
# RAG WebUI バックエンド (FastAPI)
# - Ollama API を利用した LLM 応答生成
# - RAG (Retrieval-Augmented Generation) 簡易実装
# - 数値計算デモ（将来拡張用）
# - ファイルアップロード機能
# - コード生成専用エンドポイント (/code)
# ======================================================================
#
# 2025-09 現在の仕様
# - /ask    → 通常の質問応答（RAG・一般QA）
# - /code   → コード生成専用（Python等）
# - /upload → ファイルアップロード（PDF, TXTなど）
# - /models → 利用可能なローカルモデル一覧を返す
# - Ollama の /api/generate を使用
#
# 安全対策:
# - /code では LLM が生成したコードを Python AST で検証してから返却
# - 将来的には実行前にさらに制限付きサンドボックス検証も可能
# ======================================================================

# ------------------------------------------------------
# 必要なライブラリのインポート
# ------------------------------------------------------
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import requests
import time
import shutil
import json
import ast  # Pythonコードの構文検証用
import re

# ------------------------------------------------------
# アプリケーション初期化
# ------------------------------------------------------
app = FastAPI(title="RAG WebUI Backend", version="1.0")

# フロントエンド (React / Vite) からのアクセスを許可
# 開発中なので * で全許可、本番ではドメイン制限推奨
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# 環境設定
# ------------------------------------------------------
OLLAMA_API = "http://localhost:11434/api/generate"  # Ollama サーバ API エンドポイント
UPLOAD_DIR = "./uploaded_files"  # アップロード保存先
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ------------------------------------------------------
# Ollama API 呼び出し用ヘルパー関数
# ------------------------------------------------------
def call_ollama_generate(model: str, prompt: str, stream: bool = False):
    """
    Ollama の /api/generate を呼び出して応答を取得する。
    stream=False の場合でも JSONL 形式で返ることがあるので工夫して処理する。
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.2  # デフォルトより低め → 応答安定性を重視
        }
    }
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=600)
        r.raise_for_status()

        if stream:
            # ストリーミングモード（逐次生成）
            result = ""
            for line in r.iter_lines():
                if line:
                    obj = json.loads(line.decode("utf-8"))
                    result += obj.get("response", "")
            return {"response": result}
        else:
            # 非ストリーミング → まず通常の JSON として処理
            try:
                data = r.json()
                if isinstance(data, dict):
                    return data
                elif isinstance(data, list):
                    # JSON の配列なら response を連結
                    result = "".join([d.get("response", "") for d in data])
                    return {"response": result}
            except Exception:
                # JSONL 形式を手動処理
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
# RAG 簡易実装（検索部分はダミー）
# ------------------------------------------------------
def retrieve_context(question: str):
    """
    本来はベクタDB検索（Chroma 等）で関連文書を取得する部分。
    今回はデモ用に「ダミーの関連情報」を返す。
    """
    return f"関連情報（ダミー）: {question} に関する参考資料。"


# ------------------------------------------------------
# アップロード済みファイルを RAG に組み込む処理
# ------------------------------------------------------
def load_uploaded_context():
    """
    UPLOAD_DIR に保存されたファイルをすべて読み込み
    簡易的にテキスト結合して返す。
    """
    texts = []
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        try:
            if fname.lower().endswith(".txt"):
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
            elif fname.lower().endswith(".pdf"):
                # 簡易版: PDF → テキスト (本格的には PyPDF2 や pdfplumber を使う)
                texts.append(f"[PDF: {fname} の内容はここに展開される想定]")
        except Exception as e:
            print(f"[WARN] ファイル読み込み失敗: {fname}, {e}")
            continue
    return "\n\n".join(texts)


def clean_code_output(raw: str) -> str:
    """
    LLM が返したコードから不要なマークダウンや説明を除去する。
    """
    code = raw.strip()

    # ```python ～ ``` を削除
    code = re.sub(r"^```(?:python)?", "", code, flags=re.IGNORECASE).strip()
    code = re.sub(r"```$", "", code).strip()

    # もし最初に 'python' だけの行があったら削除
    if code.splitlines()[0].strip().lower() == "python":
        code = "\n".join(code.splitlines()[1:])

    return code.strip()


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
    - RAG: アップロードファイルを毎回反映
    - Ollama から応答を取得
    """
    try:
        start = time.time()

        # アップロード済みファイルを読み込み
        file_context = load_uploaded_context()

        # ダミー検索コンテキスト + ファイル内容
        context = f"""
関連情報（ダミー）: {question} に関する参考資料。

アップロードファイルの内容:
{file_context}
"""

        # プロンプト
        prompt = f"""
以下はユーザからの質問です。関連情報を参考にして回答してください。
質問: {question}

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


def clean_code_output(raw: str) -> str:
    """
    LLM が返したコードから不要なマークダウンや説明を除去する。
    """
    if not raw:
        return ""

    code = raw.strip()

    # ```python ～ ``` を削除
    code = re.sub(r"^```(?:python)?", "", code, flags=re.IGNORECASE).strip()
    code = re.sub(r"```$", "", code).strip()

    # もし最初に 'python' だけの行があったら削除
    lines = code.splitlines()
    if lines and lines[0].strip().lower() == "python":
        code = "\n".join(lines[1:])

    return code.strip()


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

厳守事項:
- 出力は必ず Python コードのみ。
- Markdown 記法 (``` や python など) は含めない。
- 説明文・コメントも不要。
- math.pi や外部ライブラリの定数は絶対に使わない。
- エラーが出ないコードにする。
- 外部ライブラリは禁止（decimal, fractions, math.pi などは使わない）。
- 大きな整数演算は Python の組み込み int で処理すること。
- decimal.Decimal を使用する場合は必ず整数や Decimal のみを指数に使うこと。
- 平方根は (x ** 0.5) ではなく Decimal.sqrt() を使用すること。
- math や numpy の関数は使わず、標準ライブラリ decimal のみを用いること。

リクエスト:
{question}
"""

        res = call_ollama_generate(model, prompt, stream=False)
        raw_code = res.get("response", "") if res else ""

        # --- クリーニング ---
        code = clean_code_output(raw_code)

        # --- 構文チェック (安全のため) ---
        try:
            if code:
                ast.parse(code)
        except SyntaxError as e:
            return {
                "answer": f"⚠️ 生成コードに構文エラーがあります: {e}",
                "mode": "code",
                "llm_model": model,
                "sources": [],
                "elapsed": round(time.time() - start, 2),
            }

        return {
            "answer": code if code else "⚠️ コードが生成されませんでした。",
            "mode": "code",
            "llm_model": model,
            "sources": [],
            "elapsed": round(time.time() - start, 2),
        }

    except Exception as e:
        print("[ERROR] /code failed:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------
# /code エンドポイント（コード生成）
# ------------------------------------------------------
@app.post("/code")
async def generate_code(
        question: str = Form(...),
        model: str = Form("llama3.1:8b"),
):
    """
    コード生成専用エンドポイント
    - 入力質問に対して **実行可能なPythonコードのみ** を返す
    - 余計な説明やコメントは禁止
    - AST 構文チェックを行い、構文エラーは検出して返す
    """
    try:
        start = time.time()

        # コード生成用の強化プロンプト
        prompt = f"""
あなたは熟練したソフトウェアエンジニアです。
次のリクエストに対して、**実行可能なPythonコードのみ**を返してください。

厳守事項:
- 出力はPythonコードのみ
- 先頭と末尾に説明文やマークダウン記法 (``` 等) を含めない
- math.pi や外部ライブラリの定数は使わない
- 必ずエラーなく実行可能にする
- ゼロ除算やインデックスエラーは避ける
- 円周率計算なら Spigot アルゴリズムなどの整数演算を使う

リクエスト:
{question}
"""

        # Ollama API 呼び出し
        res = call_ollama_generate(model, prompt, stream=False)
        code = res.get("response", "").strip()

        # --- クリーニング ---
        code = clean_code_output(raw_code)

        # AST 構文検証
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                "answer": f"⚠️ 生成コードに構文エラーがあります: {e}",
                "mode": "code",
                "llm": "ollama",
                "llm_model": model,
                "sources": [],
                "elapsed": round(time.time() - start, 2),
            }

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
# /upload エンドポイント（ファイルアップロード）
# ------------------------------------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    ファイルアップロード処理
    - PDF, TXT などを保存する（解析処理は未実装）
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "status": "uploaded"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------
# /models エンドポイント（利用可能なモデル一覧）
# ------------------------------------------------------
@app.get("/models")
async def list_models():
    """
    利用可能なモデル一覧を返す。
    今後は Ollama CLI から自動検出する予定だが、現状は固定リスト。
    """
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
# 将来拡張メモ（コメントで行数を補完）
# ======================================================================
# - ベクタDB連携（Chroma, Weaviate など）
# - 音声入力/出力（Whisper, TTS）
# - ユーザーごとのセッション管理
# - 応答履歴の保存
# - セキュリティ強化（APIキー認証、ユーザー権限管理）
# - エラーログ・モニタリング
# ======================================================================
