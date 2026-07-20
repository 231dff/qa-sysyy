"""
智能搜索助手 — 工具函数
URL 去重、Token 计数、指数退避、限流保护
"""
from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from functools import wraps
from typing import Callable, Optional

import tiktoken

from config import PROJECT_ROOT, get_agent_config

# ============================================================
# Token 计数
# ============================================================
_encoder: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        try:
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = tiktoken.get_encoding("o200k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """估算文本 token 数量"""
    try:
        return len(_get_encoder().encode(text))
    except Exception:
        # 简单 fallback：中文约 1.5 字符/token，英文约 4 字符/token
        return len(text) // 2


def estimate_tokens(text: str) -> int:
    """count_tokens 的别名"""
    return count_tokens(text)


# ============================================================
# URL 去重 (基于滑动窗口)
# ============================================================
class URLDeduplicator:
    """
    基于哈希的 URL 去重器
    维护滑动窗口，只去重最近 N 条 URL
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._seen: list[str] = []  # 有序列表，新 URL 追加到末尾

    def is_duplicate(self, url: str) -> bool:
        """检查 URL 是否在滑动窗口中"""
        normalized = self._normalize(url)
        return normalized in self._seen

    def add(self, url: str) -> None:
        """将 URL 加入窗口；超过窗口大小时弹出最早"""
        normalized = self._normalize(url)
        if normalized in self._seen:
            self._seen.remove(normalized)
        self._seen.append(normalized)
        while len(self._seen) > self.window_size:
            self._seen.pop(0)

    def deduplicate(self, urls: list[str]) -> list[str]:
        """批量去重，返回不重复的 URL 列表"""
        result: list[str] = []
        for url in urls:
            if not self.is_duplicate(url):
                result.append(url)
                self.add(url)
        return result

    def clear(self) -> None:
        self._seen.clear()

    @staticmethod
    def _normalize(url: str) -> str:
        """URL 规范化：去尾部斜杠、统一小写"""
        url = url.strip().rstrip("/").lower()
        # 去掉常见的跟踪参数
        for param in ("utm_source", "utm_medium", "utm_campaign", "ref", "fbclid"):
            import re
            # 移除参数: 处理 ?param=val 或 &param=val
            url = re.sub(rf"[?&]{param}=[^&]*", "", url)
        # 修复因移除参数导致的符号残留
        # 情况1: "?&..." → "?..."
        url = re.sub(r"\?&", "?", url)
        # 情况2: 如果 path 后第一个字符是 & 且前面没有 ?（首参数被移除）→ 改为 ?
        url = re.sub(r"([^?])&", r"\1?", url, count=1)
        # 修复因移除尾参数导致末尾多余符号
        url = url.rstrip("?&")
        return url


# ============================================================
# 内容去重 (基于哈希)
# ============================================================
def content_fingerprint(text: str, min_length: int = 100) -> Optional[str]:
    """生成内容指纹 (SHA-256), 仅对足够长的文本"""
    if len(text) < min_length:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================
# 指数退避重试
# ============================================================
def exponential_backoff(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
):
    """
    指数退避装饰器，兼容同步和异步函数。
    用法:
        @exponential_backoff(max_retries=3)
        async def flaky_api_call(): ...

        @exponential_backoff(max_retries=3)
        def flaky_sync_call(): ...
    """
    cfg = get_agent_config()
    _max_retries = max_retries if max_retries is not None else cfg.max_retries
    _base_delay = base_delay if base_delay is not None else cfg.base_delay
    _max_delay = max_delay if max_delay is not None else cfg.max_delay

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception: Optional[Exception] = None
            for attempt in range(_max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    if attempt < _max_retries:
                        delay = min(_base_delay * (2 ** attempt), _max_delay)
                        if get_agent_config().verbose:
                            print(f"[重试] {func.__name__} 第{attempt+1}次失败 "
                                  f"({exc}), {delay:.1f}s 后重试...")
                        await asyncio.sleep(delay)
            raise last_exception  # type: ignore[misc]

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception: Optional[Exception] = None
            for attempt in range(_max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    if attempt < _max_retries:
                        delay = min(_base_delay * (2 ** attempt), _max_delay)
                        if get_agent_config().verbose:
                            print(f"[重试] {func.__name__} 第{attempt+1}次失败 "
                                  f"({exc}), {delay:.1f}s 后重试...")
                        time.sleep(delay)
            raise last_exception  # type: ignore[misc]

        import asyncio as _asyncio_mod
        if _asyncio_mod.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ============================================================
# 滑动窗口限流
# ============================================================
class RateLimiter:
    """滑动窗口限流器"""

    def __init__(self, max_per_minute: int = 30):
        self.max_per_minute = max_per_minute
        self._timestamps: list[float] = []

    def acquire(self) -> bool:
        """
        尝试获取一个请求配额
        Returns: True=放行, False=限流
        """
        now = time.time()
        window_start = now - 60.0
        # 清理过期时间戳
        self._timestamps = [t for t in self._timestamps if t > window_start]
        if len(self._timestamps) < self.max_per_minute:
            self._timestamps.append(now)
            return True
        return False

    @property
    def remaining(self) -> int:
        """剩余配额"""
        now = time.time()
        window_start = now - 60.0
        self._timestamps = [t for t in self._timestamps if t > window_start]
        return max(0, self.max_per_minute - len(self._timestamps))


# ============================================================
# 会话工具
# ============================================================
def generate_session_id() -> str:
    """生成唯一会话 ID"""
    return str(uuid.uuid4())[:8]


# ============================================================
# 上下文裁剪
# ============================================================
def trim_context(
    fragments: list[tuple[str, str, float]],  # (url, content, score)
    max_tokens: int,
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """
    双层过滤：先按分数取 Top-K，再按 token 限制裁剪
    Returns: 裁剪后的片段列表
    """
    # 第一层：Top-K 过滤
    sorted_fragments = sorted(fragments, key=lambda x: x[2], reverse=True)
    top_fragments = sorted_fragments[:top_k]

    # 第二层：Token 裁剪
    result: list[tuple[str, str, float]] = []
    total_tokens = 0
    for url, content, score in top_fragments:
        frag_tokens = count_tokens(content)
        if total_tokens + frag_tokens > max_tokens:
            # 截断最后一条
            remaining = max_tokens - total_tokens
            if remaining > 50:  # 至少保留 50 token 才有意义
                truncated = _truncate_text(content, remaining)
                result.append((url, truncated, score))
            break
        result.append((url, content, score))
        total_tokens += frag_tokens
    return result


def _truncate_text(text: str, max_tokens: int) -> str:
    """按 token 数截断文本"""
    try:
        tokens = _get_encoder().encode(text)
        if len(tokens) <= max_tokens:
            return text
        return _get_encoder().decode(tokens[:max_tokens]) + "..."
    except Exception:
        return text[: max_tokens * 2] + "..."


def format_search_context(
    results: list[tuple[str, str, float]],
) -> str:
    """将搜索结果格式化为 LLM 可读的上下文"""
    if not results:
        return "（无相关搜索结果）"

    lines: list[str] = []
    for i, (url, content, score) in enumerate(results, 1):
        lines.append(f"[来源 {i}] {url} (相关性: {score:.2f})")
        lines.append(f"{content}")
        lines.append("")
    return "\n".join(lines)
