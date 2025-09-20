import React, { useState, useRef } from "react";
import axios from "axios";

const Ask = () => {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(null);
  const [meta, setMeta] = useState({});
  const [model, setModel] = useState("mistral:7b-instruct"); // デフォルトモデル

  const askStartRef = useRef(null);

  const pushStatus = (msg) => {
    console.log("[STATUS]", msg);
  };

  const handleAsk = async () => {
    if (!question.trim()) return;

    const formData = new FormData();
    formData.append("question", question);
    formData.append("model", model); // ← 選択モデルも送信

    try {
      setLoading(true);
      setAnswer("");
      setMeta({});
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
        llm_model: res.data.llm_model || null,
        sources: Array.isArray(res.data.sources) ? res.data.sources : [],
      });

      pushStatus(
        `応答受信（mode: ${res.data.mode || "-"}, model: ${res.data.llm_model || "-"}, 時間: ${sec} 秒）`
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
    <div>
      <h2>AI 質問</h2>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="質問を入力してください"
        rows={4}
        style={{ width: "100%" }}
      />
      <br />

      {/* モデル選択ボタン群 */}
      <div style={{ margin: "10px 0" }}>
        <button onClick={() => setModel("mistral:7b-instruct")}>
          Mistral 7B
        </button>
        <button onClick={() => setModel("gemma:2b-it")}>Gemma 2B</button>
        <button onClick={() => setModel("phi3:mini")}>Phi-3 Mini</button>
        <button onClick={() => setModel("llama-2-7b-chat")}>
          LLaMA 2 7B Chat
        </button>
      </div>
      <p>🧠 現在選択中のモデル: {model}</p>

      <button onClick={handleAsk} disabled={loading}>
        {loading ? "送信中..." : "質問する"}
      </button>

      {elapsed && <p>⏱️ 応答時間: {elapsed} 秒</p>}
      {answer && <p>AIの回答: {answer}</p>}
    </div>
  );
};

export default Ask;
