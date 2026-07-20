// 智能搜索助手 — 类型定义

/** API 请求: POST /api/chat/stream */
export interface ChatRequest {
  query: string;
  session_id?: string | null;
  model?: string | null;
  search_depth?: string | null;
  top_k?: number | null;
}

/** SSE 进度事件 */
export interface ProgressEvent {
  node: "rewrite" | "search" | "generate" | "fallback" | "score";
  message: string;
}

/** SSE Token 事件 */
export interface TokenEvent {
  text: string;
}

/** 来源信息 */
export interface SourceInfo {
  url: string;
  title: string;
  snippet: string;
}

/** SSE 来源事件 */
export interface SourcesEvent {
  sources: SourceInfo[];
}

/** SSE 完成事件 */
export interface DoneEvent {
  confidence: number;
  latency_ms: number;
  tokens_used: number;
  is_fallback: boolean;
}

/** SSE 错误事件 */
export interface ErrorEvent {
  message: string;
  code: string;
}

/** 联合 SSE 事件 */
export type SSEEvent =
  | { type: "progress"; data: ProgressEvent }
  | { type: "token"; data: TokenEvent }
  | { type: "sources"; data: SourcesEvent }
  | { type: "done"; data: DoneEvent }
  | { type: "error"; data: ErrorEvent };

/** 聊天消息 */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceInfo[];
  meta?: {
    confidence: number;
    latency_ms: number;
    tokens_used: number;
    is_fallback: boolean;
  };
}

/** 会话信息 */
export interface SessionInfo {
  session_id: string;
  history: { role: string; content: string }[];
  active: boolean;
}

/** 默认配置 */
export interface ConfigDefaults {
  model: string;
  search_depth: string;
  top_k: number;
  llm_api_base: string;
}

/** 用户设置 */
export interface UserSettings {
  model: string;
  searchDepth: string;
  topK: number;
}
