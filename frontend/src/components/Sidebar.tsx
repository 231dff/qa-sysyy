"use client";

// 智能搜索助手 — 侧边栏: 设置 + 会话管理
import { useChat } from "../providers/ChatProvider";

export default function Sidebar() {
  const { settings, updateSettings, clearHistory, newSession, sessionId, sse } = useChat();

  const isBusy = sse.status === "connecting" || sse.status === "streaming";

  return (
    <aside className="w-80 h-screen bg-white border-r border-gray-200 flex flex-col p-4 shrink-0">
      {/* 标题 */}
      <h1 className="text-xl font-bold text-gray-800 mb-1">🔍 智能搜索助手</h1>
      <p className="text-xs text-gray-400 mb-4">实时联网搜索 · 流式问答</p>
      <hr className="mb-4" />

      {/* 设置 */}
      <section className="mb-4">
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-2">⚙️ 设置</h2>

        <label className="block text-xs text-gray-600 mb-1">LLM 模型</label>
        <input
          type="text"
          value={settings.model}
          onChange={(e) => updateSettings({ model: e.target.value })}
          disabled={isBusy}
          placeholder="输入模型名称..."
          className="w-full text-sm border rounded px-2 py-1.5 mb-3 bg-white disabled:opacity-50"
        />

        <label className="block text-xs text-gray-600 mb-1">搜索深度</label>
        <div className="flex gap-2 mb-3">
          <button
            onClick={() => updateSettings({ searchDepth: "basic" })}
            disabled={isBusy}
            className={`flex-1 text-sm border rounded px-2 py-1 ${
              settings.searchDepth === "basic"
                ? "bg-blue-50 border-blue-300 text-blue-700"
                : "bg-white text-gray-600"
            } disabled:opacity-50`}
          >
            Basic
          </button>
          <button
            onClick={() => updateSettings({ searchDepth: "advanced" })}
            disabled={isBusy}
            className={`flex-1 text-sm border rounded px-2 py-1 ${
              settings.searchDepth === "advanced"
                ? "bg-blue-50 border-blue-300 text-blue-700"
                : "bg-white text-gray-600"
            } disabled:opacity-50`}
          >
            Advanced
          </button>
        </div>

        <label className="block text-xs text-gray-600 mb-1">
          Top-K 片段: <span className="font-semibold">{settings.topK}</span>
        </label>
        <input
          type="range"
          min={2}
          max={10}
          value={settings.topK}
          onChange={(e) => updateSettings({ topK: Number(e.target.value) })}
          disabled={isBusy}
          className="w-full mb-4 disabled:opacity-50"
        />
      </section>

      <hr className="mb-4" />

      {/* 会话操作 */}
      <section className="mb-4 flex gap-2">
        <button
          onClick={newSession}
          className="flex-1 text-sm bg-blue-500 hover:bg-blue-600 text-white rounded px-3 py-2 transition"
        >
          🆕 新会话
        </button>
        <button
          onClick={clearHistory}
          className="flex-1 text-sm bg-gray-200 hover:bg-gray-300 text-gray-700 rounded px-3 py-2 transition"
        >
          🗑️ 清历史
        </button>
      </section>

      {/* 会话 ID */}
      <div className="text-xs text-gray-400 truncate">
        会话: <code className="bg-gray-100 px-1 py-0.5 rounded">{sessionId}</code>
      </div>

      {/* Footer */}
      <div className="mt-auto pt-4">
        <p className="text-xs text-gray-300 text-center">
          Powered by FastAPI + SSE + Next.js
        </p>
      </div>
    </aside>
  );
}
