import React, { useState, useRef } from "react";
import axios from "axios";

export default function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(null);
  const [meta, setMeta] = useState({ mode: null, llm: null, sources: [] });

  // 状態ログ
  const [statusLog, setStatusLog] = useState([]);
  const askStartRef = useRef(null);

  const pushStatus = (msg) => {
    const t = new Date().toLocaleTimeString();
    setStatusLog((prev) => [...prev, `[${t}] ${msg}`]);
  };

  // ファイルアップロード
  const handleUpload = async () => {
    if (!file) {
      alert("アップロードするファイルを選択してください");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

    try {
      pushStatus("アップロード開始");
      const res = await axios.post("http://127.0.0.1:8000/upload", formData);
      pushStatus(
        `アップロード完了: ${res.data.uploaded}（チャンク数: ${res.data.chunks}）`
      );
      alert("アップロード完了: " + res.data.uploaded);
    } catch (err) {
      console.error(err);
      pushStatus("アップロードに失敗しました");
      alert("アップロードに失敗しました");
    }
  };

  // 質問送信
  const handleAsk = async () => {
    if (!question.trim()) return;

    const formData = new FormData();
    formData.append("question", question);

    try {
      setLoading(true);
      setAnswer("");
      setMeta({ mode: null, llm: null, sources: [] });
      setElapsed(null);
      pushStatus("質問送信");

      askStartRef.current = performance.now();

      const res = await axios.post("http://127.0.0.1:8000/ask", formData);

      const end = performance.now();
      const sec = ((end - askStartRef.current) / 1000).toFixed(2);
      setElapsed(sec);

      setAnswer(res.data.answer || "");
      setMeta({
        mode: res.data.mode || null,
        llm: res.data.llm || null,
        sources: Array.isArray(res.data.sources) ? res.data.sources : [],
      });

      pushStatus(
        `応答受信（mode: ${res.data.mode || "-"}, llm: ${
          res.data.llm || "-"
        }, 時間: ${sec} 秒）`
      );
    } catch (err) {
      console.error(err);
      setAnswer("⚠️ エラーが発生しました");
      pushStatus("質問処理でエラー発生");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-3xl mx-auto font-sans space-y-6">
      <h1 className="text-3xl font-bold text-blue-600">📚 RAG WebUI</h1>

      {/* アップロード行 */}
      <div className="flex items-center space-x-4">
        <input
          id="fileInput"
          type="file"
          className="hidden"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <label
          htmlFor="fileInput"
          className="px-4 h-[42px] flex items-center bg-gray-500 text-white rounded cursor-pointer hover:bg-gray-600"
        >
          ファイルを選択
        </label>
        <span className="truncate max-w-[400px]">
          {file ? file.name : "（未選択）"}
        </span>
        <div className="flex-1" />
        <button
          onClick={handleUpload}
          className="px-4 h-[42px] bg-green-500 text-white rounded hover:bg-green-600"
        >
          アップロード
        </button>
      </div>

      {/* 質問入力 */}
      <div className="space-y-2">
        <textarea
          className="w-full border p-3 rounded"
          rows="3"
          placeholder="質問を入力してください"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <div className="flex items-center gap-3">
          <button
            onClick={handleAsk}
            disabled={loading}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
          >
            {loading ? "処理中..." : "質問する"}
          </button>

          {/* くるくるマーク */}
          {loading && (
            <div className="flex items-center gap-2 text-gray-600">
              <svg
                className="animate-spin h-5 w-5"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                />
              </svg>
              <span>サーバー処理中…</span>
            </div>
          )}
        </div>
      </div>

      {/* 回答表示 */}
      {answer && (
        <div className="p-4 border rounded bg-gray-50">
          <h2 className="font-semibold mb-2 text-lg">AIの回答:</h2>
          <pre className="whitespace-pre-wrap font-sans text-lg leading-relaxed">
            {answer}
          </pre>
          <div className="mt-3 text-sm text-gray-600 space-y-1">
            {elapsed && <div>⏱️ 回答時間: {elapsed} 秒</div>}
            {meta.mode && <div>🧭 モード: {meta.mode}</div>}
            {meta.llm && <div>🧠 モデル: {meta.llm}</div>}
            {!!meta.sources?.length && (
              <div>
                📎 出典:
                <ul className="list-disc ml-6">
                  {meta.sources.map((s, i) => (
                    <li key={`${s.source}-${s.page}-${i}`}>
                      {s.source}
                      {s.page != null ? `（p.${s.page}）` : ""}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 状態ログ */}
      <div className="p-4 border rounded bg-white">
        <h3 className="font-semibold mb-2">🛠️ 動作の状態</h3>
        {statusLog.length === 0 ? (
          <p className="text-sm text-gray-500">まだイベントはありません。</p>
        ) : (
          <ul className="text-sm space-y-1 max-h-48 overflow-auto">
            {statusLog.map((line, idx) => (
              <li key={idx} className="whitespace-pre-wrap">
                {line}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}