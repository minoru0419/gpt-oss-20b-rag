// frontend/src/App.jsx
// ======================================================================
// フロントエンド (React + Vite)
// - モデル選択プルダウン
// - 質問入力フォーム
// - 通常送信ボタン（/ask）
// - コード生成ボタン（/code）
// - ファイル選択 + アップロードボタン（/upload → /ask に反映）
// - くるくるマーク付き
// - 回答表示エリア
// - ステータスログ
// ======================================================================

import React, { useState, useRef } from "react";
import axios from "axios";

function App() {
  // 入力中の質問
  const [question, setQuestion] = useState("");

  // モデル選択（デフォルトは llama3.1:8b）
  const [model, setModel] = useState("llama3.1:8b");

  // API 応答
  const [answer, setAnswer] = useState("");

  // メタ情報
  const [meta, setMeta] = useState({
    mode: null,
    llm: null,
    llm_model: null,
    sources: [],
  });

  // 経過時間
  const [elapsed, setElapsed] = useState(null);

  // ローディング状態（スピナー）
  const [loading, setLoading] = useState(false);

  // ステータスログ
  const [statusLog, setStatusLog] = useState([]);

  // ファイル選択
  const [selectedFile, setSelectedFile] = useState(null);

  // 処理開始時間計測用
  const askStartRef = useRef(null);

  // ステータスを追加
  const pushStatus = (msg) => {
    setStatusLog((prev) => [
      ...prev,
      `[${new Date().toLocaleTimeString()}] ${msg}`,
    ]);
  };

  // 共通の送信処理（/ask or /code）
  const handleSend = async (endpoint) => {
    if (!question.trim()) return;

    const formData = new FormData();
    formData.append("question", question);
    formData.append("model", model);

    try {
      setLoading(true);
      setAnswer("");
      setMeta({ mode: null, llm: null, llm_model: null, sources: [] });
      setElapsed(null);
      pushStatus(`質問送信 → ${endpoint}`);

      askStartRef.current = performance.now();

      const res = await axios.post(`http://127.0.0.1:8000${endpoint}`, formData);

      const end = performance.now();
      const sec = ((end - askStartRef.current) / 1000).toFixed(2);
      setElapsed(sec);

      setAnswer(res.data.answer || "");
      setMeta({
        mode: res.data.mode || null,
        llm: res.data.llm || null,
        llm_model: res.data.llm_model || null,
        sources: Array.isArray(res.data.sources) ? res.data.sources : [],
      });

      pushStatus(
        `応答受信 (mode: ${res.data.mode || "-"}, model: ${
          res.data.llm_model || "-"
        }, 時間: ${sec} 秒)`
      );
    } catch (err) {
      console.error(err);
      setAnswer("⚠️ エラーが発生しました");
      pushStatus("質問処理でエラー発生");
    } finally {
      setLoading(false);
    }
  };

  // ファイルアップロード処理
  const handleUpload = async () => {
    if (!selectedFile) {
      alert("ファイルを選択してください");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      setLoading(true);
      pushStatus(`ファイルアップロード: ${selectedFile.name}`);

      const res = await axios.post("http://127.0.0.1:8000/upload", formData);

      pushStatus(`アップロード完了: ${res.data.filename}`);
      alert("アップロードが完了しました。このファイルが回答に利用されます。");
    } catch (err) {
      console.error(err);
      pushStatus("アップロード失敗");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center p-6">
      {/* タイトル */}
      <h1 className="text-3xl font-bold text-gray-800 mb-6">
        🚀 RAG WebUI (Ollama API)
      </h1>

      {/* 入力フォーム */}
      <div className="w-full max-w-3xl bg-white rounded-xl shadow p-6 space-y-4">
        {/* モデル選択プルダウン */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            使用モデル
          </label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full border rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="mistral:7b-instruct">Mistral 7B — 和文安定</option>
            <option value="qwen2.5:3b-instruct">Qwen2.5 3B — 軽量高速</option>
            <option value="phi3:mini">Phi-3 Mini — 超軽量</option>
            <option value="gemma:2b-instruct">Gemma 2B — 軽い</option>
            <option value="tinyllama:chat">TinyLLaMA Chat — テスト用</option>
            <option value="neural-chat:7b">Neural Chat 7B — チャット向け</option>
            <option value="qwen2:7b-instruct">Qwen2 7B — 新系7B</option>
            <option value="llama2:7b-chat">LLaMA 2 7B Chat — 枯れて安定</option>
            <option value="llama3.1:8b">LLaMA 3.1 8B — 最新・コード得意</option>
          </select>
        </div>

        {/* 質問入力 */}
        <textarea
          className="w-full border rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows="3"
          placeholder="質問を入力してください..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />

        {/* 送信ボタン群 */}
        <div className="flex space-x-4">
          <button
            onClick={() => handleSend("/ask")}
            disabled={loading}
            className={`flex-1 py-3 rounded-lg font-semibold text-white flex justify-center items-center space-x-2 transition ${
              loading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-blue-500 hover:bg-blue-600"
            }`}
          >
            {loading && (
              <svg
                className="animate-spin h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                ></path>
              </svg>
            )}
            <span>{loading ? "送信中..." : "送信 🚀"}</span>
          </button>

          <button
            onClick={() => handleSend("/code")}
            disabled={loading}
            className={`flex-1 py-3 rounded-lg font-semibold text-white flex justify-center items-center transition ${
              loading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-green-500 hover:bg-green-600"
            }`}
          >
            💻 コード生成
          </button>
        </div>

        {/* ファイルアップロード */}
        <div className="flex space-x-4">
          <input
            type="file"
            onChange={(e) => setSelectedFile(e.target.files[0])}
            className="flex-1 py-3 px-3 rounded-lg bg-blue-100 text-gray-800 border border-blue-300 cursor-pointer hover:bg-blue-200"
          />
          <button
            onClick={handleUpload}
            disabled={loading}
            className={`py-3 px-6 rounded-lg font-semibold text-white flex justify-center items-center transition ${
              loading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-purple-500 hover:bg-purple-600"
            }`}
          >
            ⬆️ アップロード
          </button>
        </div>
      </div>

      {/* 回答エリア */}
      <div className="w-full max-w-3xl bg-white rounded-xl shadow p-6 mt-6">
        <h2 className="text-lg font-semibold mb-2">AIの回答:</h2>
        <div className="whitespace-pre-wrap text-gray-800">{answer}</div>
        {elapsed && (
          <p className="text-sm text-gray-500 mt-2">⏱️ {elapsed} 秒</p>
        )}
      </div>

      {/* ステータスログ */}
      <div className="w-full max-w-3xl bg-gray-100 rounded-xl shadow p-4 mt-6">
        <h2 className="text-lg font-semibold mb-2">ステータス</h2>
        <div className="text-sm text-gray-600 space-y-1 max-h-40 overflow-y-auto">
          {statusLog.map((line, idx) => (
            <div key={idx}>{line}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
