"""
智能搜索助手 — FastAPI 后端
提供 SSE 流式聊天端点 + 会话管理 REST API
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from agent.graph import SearchAgent
from agent.streaming import run_stream_sse
from config import get_api_config, get_agent_config
from memory.session_store import get_session_store
from utils.helpers import generate_session_id

app = FastAPI(
    title="智能搜索助手 API",
    description="实时联网搜索 Agent 的 FastAPI 后端服务",
    version="1.0.0",
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://frontend:3000",   # Docker 内部网络
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 全局单例 ──
_agent: Optional[SearchAgent] = None


def get_agent() -> SearchAgent:
    global _agent
    if _agent is None:
        _agent = SearchAgent()
    return _agent


# ============================================================
# Pydantic 模型
# ============================================================
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话 ID (不传则自动生成)")
    model: Optional[str] = Field(None, description="LLM 模型覆盖 (e.g. gpt-4o)")
    search_depth: Optional[str] = Field(None, description="搜索深度 (basic | advanced)")
    top_k: Optional[int] = Field(None, ge=2, le=10, description="保留 Top-K 片段")


class SessionInfo(BaseModel):
    session_id: str
    history: list[dict]
    active: bool


class ConfigDefaults(BaseModel):
    model: str
    search_depth: str
    top_k: int
    llm_api_base: str


class HealthResponse(BaseModel):
    status: str
    active_sessions: int


# ============================================================
# SSE 聊天端点
# ============================================================
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """SSE 流式聊天 — 逐 token 推送答案生成进度"""
    agent = get_agent()
    sid = req.session_id or generate_session_id()

    async def generate():
        async for sse_data in run_stream_sse(
            agent=agent,
            user_query=req.query,
            session_id=sid,
            model_override=req.model,
            search_depth_override=req.search_depth,
            top_k_override=req.top_k,
        ):
            # 客户端断开时停止
            if await request.is_disconnected():
                break
            yield sse_data.encode("utf-8")
            await asyncio.sleep(0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 会话管理端点
# ============================================================
@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    agent = get_agent()
    history = agent.get_history(session_id)
    active = session_id in agent._sessions
    return SessionInfo(session_id=session_id, history=history, active=active)


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    agent = get_agent()
    agent.clear_session(session_id)
    return {"ok": True}


# ============================================================
# 配置端点
# ============================================================
@app.get("/api/config/defaults")
async def get_config_defaults():
    api_cfg = get_api_config()
    agent_cfg = get_agent_config()
    return ConfigDefaults(
        model=api_cfg.llm_model,
        search_depth=agent_cfg.search_depth,
        top_k=agent_cfg.top_k_fragments,
        llm_api_base=api_cfg.llm_api_base,
    )


# ============================================================
# 健康检查
# ============================================================
@app.get("/api/health")
async def health():
    agent = get_agent()
    return HealthResponse(
        status="ok",
        active_sessions=len(agent._sessions),
    )


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
