// 智能搜索助手 — API 客户端
import type { SessionInfo, ConfigDefaults } from "./types";

const BASE = ""; // 同域，通过 Next.js API Route 代理或 rewrites

/** GET /api/session/:id — 获取会话历史 */
export async function getSession(
  sessionId: string
): Promise<SessionInfo> {
  const res = await fetch(`${BASE}/api/session/${sessionId}`);
  if (!res.ok) throw new Error(`获取会话失败: ${res.status}`);
  return res.json();
}

/** DELETE /api/session/:id — 清除会话 */
export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/api/session/${sessionId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`删除会话失败: ${res.status}`);
}

/** GET /api/config/defaults — 获取默认配置 */
export async function getConfigDefaults(): Promise<ConfigDefaults> {
  const res = await fetch(`${BASE}/api/config/defaults`);
  if (!res.ok) throw new Error(`获取配置失败: ${res.status}`);
  return res.json();
}

/** 生成随机会话 ID */
export function generateSessionId(): string {
  return `s_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}
