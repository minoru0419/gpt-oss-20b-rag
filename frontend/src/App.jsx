import React, { useState, useRef } from "react";
import axios from "axios";

export default function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(null);
  const [sources, setSources] = useState([]);

  const askStartRef = useRef(null);

  // ファイルアップロード
  const handleUpload = async () => {
    if (!file) {
      alert("アップロードするファイルを選択してください");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://127.0.0.1:8000/upload", formData);
      alert(`アップロード完了: ${res.data.uploaded} (chunks=${res.data.chunks})`);
    } catch (err) {
      console.error(err);
      alert("アップロード失敗: " + (err.response?.data?.error || err.message));
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
      setSources([]);
      setElapsed(null);
      askStartRef.current = performance.now();

      const res = await axios.post("http://127.0.0.1:8000/ask", formData);

      const sec = ((performance.now() - askStartRef.current) / 1000).toFixed(2);
      setElapsed(sec);
      setAnswer(res.data.answer || "");
      setSources(res.data.sources || []);

    } catch (err) {
      console.error(err);
      let msg = "⚠️ エラーが発生しました";
      if (err.response?.data?.error) {
        msg = `⚠️ サーバーエラー: ${err.response.data.error}`;
      } else if (err.message) {
        msg = `⚠️ クライアントエラー: ${err.message}`;
      }
      setAnswer(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-3xl mx-auto font-sans space-y-6">
      <h1 className="text-3xl font-bold text-blue-600">📚 RAG WebUI</h1>

      {/* ファイルアップロード */}
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
        <span>{file ? file.name : "（未選択）"}</span>
        <button
          onClick={handleUpload}
          className="px-4 h-[42px] bg-green-500 text-white rounded hover:bg-green-600"
        >
          アップロード
        </button>
      </div>

      {/* 質問入力 */}
      <div>
        <textarea
          className="w-full border p-3 rounded"
          rows="3"
          placeholder="質問を入力してください"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button
          onClick={handleAsk}
          disabled={loading}
          className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? "処理中..." : "質問する"}
        </button>
      </div>

      {/* 回答表示 */}
      {answer && (
        <div className="p-4 border rounded bg-gray-50">
          <h2 className="font-semibold mb-2">AIの回答:</h2>
          <pre className="whitespace-pre-wrap">{answer}</pre>
          {elapsed && <div>⏱️ 回答時間: {elapsed} 秒</div>}
          {sources.length > 0 && (
            <div>
              📎 出典:
              <ul className="list-disc ml-6">
                {sources.map((s, i) => (
                  <li key={i}>{s.source} (chunk {s.chunk_id})</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
