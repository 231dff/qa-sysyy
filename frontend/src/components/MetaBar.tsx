"use client";

// 智能搜索助手 — 元信息栏 (置信度 / 延迟 / Token / 降级)
import type { DoneEvent } from "../lib/types";

export default function MetaBar({ meta }: { meta: DoneEvent }) {
  const icon = meta.is_fallback ? "⚠️" : "✅";
  const confidencePct = `${Math.round(meta.confidence * 100)}%`;

  return (
    <p className="text-xs text-gray-400 mt-1">
      {icon} 置信度: {confidencePct} | 耗时: {meta.latency_ms.toFixed(0)}ms | Token: {meta.tokens_used}
      {meta.is_fallback && (
        <span className="ml-2 px-1.5 py-0.5 bg-orange-100 text-orange-700 rounded text-[10px] font-semibold">
          降级回答
        </span>
      )}
    </p>
  );
}
