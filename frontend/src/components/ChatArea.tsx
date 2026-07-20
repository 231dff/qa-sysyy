"use client";

// 智能搜索助手 — 聊天区域: 消息列表 + 输入框 + 流式实时消息
import { useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";
import ChatInput from "./ChatInput";
import { useChat } from "../providers/ChatProvider";
import type { ChatMessage } from "../lib/types";

export default function ChatArea() {
  const { messages, sendMessage, sse } = useChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  const isBusy = sse.status === "connecting" || sse.status === "streaming";

  // 自动滚动到底部
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sse.tokens]);

  // 构建流式中的临时消息 — 只在流式进行中（connecting/streaming）显示，done 后的消息由 ChatProvider 加入 messages
  const streamingMsg: ChatMessage | null =
    isBusy
      ? {
          id: "streaming",
          role: "assistant" as const,
          content: sse.tokens,
          sources: sse.sources.length > 0 ? sse.sources : undefined,
          meta: sse.meta ?? undefined,
        }
      : null;

  const displayMessages = streamingMsg
    ? [...messages, streamingMsg]
    : messages;

  return (
    <div className="flex-1 flex flex-col h-screen bg-gray-50">
      {/* 状态栏 */}
      {sse.progressMessage && isBusy && (
        <div className="bg-blue-50 border-b border-blue-100 px-4 py-2">
          <p className="text-xs text-blue-600 flex items-center gap-2">
            <span className="inline-block w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
            {sse.progressMessage}
          </p>
        </div>
      )}

      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        <div className="max-w-3xl mx-auto">
          {displayMessages.length === 0 && !isBusy && (
            <div className="text-center text-gray-400 mt-20">
              <p className="text-4xl mb-4">🔍</p>
              <p className="text-lg font-medium">智能搜索助手</p>
              <p className="text-sm">实时联网搜索 · 可信答案生成 · 多轮对话理解</p>
            </div>
          )}
          {displayMessages.map((msg) => (
            <MessageBubble
              key={msg.id}
              msg={msg}
              isStreaming={msg.id === "streaming" && isBusy}
            />
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* 输入框 */}
      <ChatInput onSubmit={sendMessage} disabled={isBusy} />
    </div>
  );
}
