"use client";

// 智能搜索助手 — SSE 流式消费 Hook
// 注意: SSE 流式请求必须直连后端，不能走 Next.js rewrites 代理（会被缓冲，导致前端收不到 token）
import { useState, useCallback, useRef } from "react";
import type { ChatRequest, SSEEvent, SourceInfo, DoneEvent } from "../lib/types";

/** SSE 流式端点 — 直连后端。
 *  - 本地开发: http://localhost:8000
 *  - Docker/生产: Next.js API Route 代理 (同域，无跨域)
 *  Docker 时前端不存在跨域问题 (rewrites)，所以走相对路径即可
 */
const SSE_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface UseSSEState {
  status: "idle" | "connecting" | "streaming" | "done" | "error";
  tokens: string;
  sources: SourceInfo[];
  meta: DoneEvent | null;
  error: string | null;
  progressNode: string | null;
  progressMessage: string | null;
}

export function useSSE() {
  const [state, setState] = useState<UseSSEState>({
    status: "idle",
    tokens: "",
    sources: [],
    meta: null,
    error: null,
    progressNode: null,
    progressMessage: null,
  });
  const abortRef = useRef<AbortController | null>(null);

  const startStream = useCallback(async (request: ChatRequest) => {
    // 取消上一次请求
    if (abortRef.current) {
      abortRef.current.abort();
    }

    const controller = new AbortController();
    abortRef.current = controller;

    setState({
      status: "connecting",
      tokens: "",
      sources: [],
      meta: null,
      error: null,
      progressNode: null,
      progressMessage: null,
    });

    try {
      const res = await fetch(`${SSE_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`${res.status}: ${errText}`);
      }

      const reader = res.body?.getReader();
      if (!reader) {
        throw new Error("无法读取响应流");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      setState((prev) => ({ ...prev, status: "streaming" }));

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        // 最后一个不完整行留在 buffer 中
        buffer = lines.pop() || "";

        let currentEvent = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith("data: ") && currentEvent) {
            try {
              const data = JSON.parse(line.slice(6));
              handleEvent(currentEvent, data, setState);
            } catch {
              // JSON 解析失败，跳过
            }
            currentEvent = "";
          }
        }
      }

      setState((prev) => {
        if (prev.status !== "error") {
          return { ...prev, status: "done" };
        }
        return prev;
      });
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setState((prev) => ({
        ...prev,
        status: "error",
        error: (err as Error).message,
      }));
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState({
      status: "idle",
      tokens: "",
      sources: [],
      meta: null,
      error: null,
      progressNode: null,
      progressMessage: null,
    });
  }, []);

  return { ...state, startStream, cancel, reset };
}

/** 将 SSE 事件分发到 setState 更新 */
function handleEvent(
  eventType: string,
  data: any,
  setState: React.Dispatch<React.SetStateAction<UseSSEState>>
) {
  switch (eventType) {
    case "progress":
      setState((prev) => ({
        ...prev,
        progressNode: data.node,
        progressMessage: data.message,
      }));
      break;
    case "token":
      setState((prev) => ({ ...prev, tokens: prev.tokens + data.text }));
      break;
    case "sources":
      setState((prev) => ({ ...prev, sources: data.sources }));
      break;
    case "done":
      setState((prev) => ({ ...prev, meta: data }));
      break;
    case "error":
      setState((prev) => ({
        ...prev,
        status: "error",
        error: data.message,
      }));
      break;
  }
}
