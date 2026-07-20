"use client";

// 智能搜索助手 — 流式打字效果文本
import { useEffect, useState } from "react";

export default function StreamingToken({
  text,
  isStreaming,
}: {
  text: string;
  isStreaming: boolean;
}) {
  const [showCursor, setShowCursor] = useState(true);

  useEffect(() => {
    if (!isStreaming) {
      setShowCursor(false);
      return;
    }
    const timer = setInterval(() => {
      setShowCursor((v) => !v);
    }, 500);
    return () => clearInterval(timer);
  }, [isStreaming]);

  return (
    <span>
      {text}
      {showCursor && isStreaming && (
        <span className="inline-block w-0.5 h-4 bg-blue-500 ml-0.5 animate-pulse" />
      )}
    </span>
  );
}
