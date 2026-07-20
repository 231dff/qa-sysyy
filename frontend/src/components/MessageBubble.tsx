"use client";

// 智能搜索助手 — 单条聊天消息
import MarkdownContent from "./MarkdownContent";
import SourcesPanel from "./SourcesPanel";
import MetaBar from "./MetaBar";
import type { ChatMessage } from "../lib/types";

export default function MessageBubble({
  msg,
  isStreaming,
}: {
  msg: ChatMessage;
  isStreaming: boolean;
}) {
  const isUser = msg.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-blue-500 text-white rounded-br-md"
            : "bg-white border border-gray-200 shadow-sm rounded-bl-md"
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
        ) : (
          <>
            <MarkdownContent text={msg.content} isStreaming={isStreaming} />
            {msg.sources && msg.sources.length > 0 && (
              <SourcesPanel sources={msg.sources} />
            )}
            {msg.meta && <MetaBar meta={msg.meta} />}
          </>
        )}
      </div>
    </div>
  );
}
