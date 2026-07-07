"""
智能搜索助手 — 会话内存
基于会话 ID 的上下文缓存，支持过期清理与跨轮意图连贯
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from config import get_agent_config


class SessionStore:
    """
    线程安全的会话存储
    - 每个 session_id 对应一组对话历史
    - 支持 TTL 过期自动清理
    """

    def __init__(self, ttl_seconds: Optional[int] = None):
        self._ttl = ttl_seconds or get_agent_config().session_ttl_seconds
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[dict[str, str]]:
        """获取会话历史 (自动续期)"""
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return []
            entry["last_access"] = time.time()
            return entry["history"]

    def put(self, session_id: str, history: list[dict[str, str]]) -> None:
        """存储/更新会话历史"""
        with self._lock:
            self._store[session_id] = {
                "history": history,
                "created_at": self._store.get(session_id, {}).get("created_at", time.time()),
                "last_access": time.time(),
            }

    def delete(self, session_id: str) -> None:
        """删除会话"""
        with self._lock:
            self._store.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        with self._lock:
            return session_id in self._store

    def cleanup_expired(self) -> int:
        """清理过期会话，返回清理数量"""
        now = time.time()
        expired = []
        with self._lock:
            for sid, entry in self._store.items():
                if now - entry["last_access"] > self._ttl:
                    expired.append(sid)
            for sid in expired:
                del self._store[sid]
        return len(expired)

    @property
    def active_sessions(self) -> int:
        """活跃会话数"""
        with self._lock:
            return len(self._store)

    def list_sessions(self) -> list[str]:
        """列出所有会话 ID"""
        with self._lock:
            return list(self._store.keys())


# 全局单例
_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
