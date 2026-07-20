"use client";

// 智能搜索助手 — 来源面板
import type { SourceInfo } from "../lib/types";

export default function SourcesPanel({ sources }: { sources: SourceInfo[] }) {
  if (!sources || sources.length === 0) return null;

  return (
    <details className="mt-2">
      <summary className="text-xs text-blue-500 cursor-pointer hover:text-blue-700">
        📚 参考来源 ({sources.length})
      </summary>
      <ul className="mt-2 space-y-1 border-t border-gray-100 pt-2">
        {sources.map((s, i) => (
          <li key={i} className="text-xs">
            <strong>[{i + 1}]</strong>{" "}
            <a
              href={s.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:underline break-all"
            >
              {s.title || s.url}
            </a>
            {s.snippet && (
              <p className="text-gray-400 mt-0.5 ml-4 line-clamp-2">
                {s.snippet}
              </p>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}
