"""
共享 HTTP 客户端 — 连接池复用
消除每次 API 调用的 TCP/TLS 握手开销 (~200-500ms × 每次调用)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

import httpx

_shared_client: Optional[httpx.AsyncClient] = None


def set_shared_client(client: httpx.AsyncClient) -> None:
    """注入全局共享客户端 (在 server.py startup 时调用)"""
    global _shared_client
    _shared_client = client


def reset_shared_client() -> None:
    """关闭并重置共享客户端 (用于测试清理)"""
    global _shared_client
    if _shared_client is not None:
        # 安排关闭但不 await — 调用方负责
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_shared_client.aclose())
        except RuntimeError:
            pass
    _shared_client = None


@asynccontextmanager
async def get_http_client(timeout: float = 30.0):
    """
    获取 HTTP 客户端。

    如果已设置共享客户端 (生产模式)，直接复用，消除握手开销。
    否则创建临时客户端 (测试/开发兼容模式)。

    用法:
        async with get_http_client(timeout=30.0) as client:
            resp = await client.post(url, ...)
    """
    if _shared_client is not None:
        yield _shared_client
    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            yield client
