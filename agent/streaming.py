"""
智能搜索助手 — SSE 流式传输编排器
按 Agent 流水线的每个节点边界发射细粒度事件，向前端推送实时进度。
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Callable, Optional

from agent.graph import (
    SearchAgent,
    node_error_recovery,
    node_generate_answer,
    node_rewrite_query,
    node_search,
    route_after_search,
)
from agent.models import AgentState, GeneratedAnswer
from config import get_agent_config
from utils.helpers import generate_session_id
from datetime import datetime, timezone


# ============================================================
# SSE 事件格式工具
# ============================================================
def _sse_event(event: str, data: dict) -> str:
    """将 Python dict 格式化为 SSE 兼容的字节流行。"""
    import json

    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


# ============================================================
# 核心: run_stream_sse — 真正实时逐 token 推送
# ============================================================
async def run_stream_sse(
    agent: SearchAgent,
    user_query: str,
    session_id: Optional[str] = None,
    model_override: Optional[str] = None,
    search_depth_override: Optional[str] = None,
    top_k_override: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """
    执行 Agent 流水线并实时推送 SSE 事件。

    通过 asyncio.Queue 实现生产者-消费者模式：
    - 生产者：LLM 流式回调，每收到一个 token 就放入队列
    - 消费者：本生成器，从队列取出 token 并立即 yield SSE 事件
    用户无需等待生成完毕即可看到逐字输出。

    Yields:
        SSE 格式字符串，每个 yield 是 "event: ...\ndata: {...}\n\n"
    """
    agent_cfg = get_agent_config()
    from config import get_api_config

    api_cfg = get_api_config()

    # ── 运行时配置覆盖 ──
    orig_model = api_cfg.llm_model
    orig_depth = agent_cfg.search_depth
    orig_top_k = agent_cfg.top_k_fragments

    if model_override:
        api_cfg.llm_model = model_override
    if search_depth_override:
        agent_cfg.search_depth = search_depth_override
    if top_k_override is not None:
        agent_cfg.top_k_fragments = top_k_override

    try:
        # ── 会话初始化 ──
        if session_id is None:
            session_id = generate_session_id()
        if session_id not in agent._sessions:
            agent._sessions[session_id] = []
        history = agent._sessions[session_id]

        # ── 构建 AgentState ──
        state: AgentState = {
            "session_id": session_id,
            "user_query": user_query,
            "user_query_raw": user_query,
            "rewritten_queries": [],
            "search_results": [],
            "deduped_results": [],
            "relevance_scores": None,
            "top_results": [],
            "final_answer": None,
            "history": history,
            "error": None,
            "retry_count": 0,
            "fallback_triggered": False,
            "search_failed": False,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": "",
        }

        # ── Step 1: 查询改写 ──
        yield _sse_event(
            "progress",
            {"node": "rewrite", "message": f"正在改写查询: {user_query[:60]}..."},
        )
        state.update(await node_rewrite_query(state))

        # ── Step 2: 实时搜索 ──
        yield _sse_event(
            "progress",
            {"node": "search", "message": "正在搜索相关信息..."},
        )
        state.update(await node_search(state))

        # ── Step 3: 路由 + 答案生成 (实时流式) ──
        route = route_after_search(state)
        if route == "error":
            yield _sse_event(
                "progress",
                {
                    "node": "fallback",
                    "message": "搜索未获取到有效结果，切换为降级回答...",
                },
            )
            state.update(await node_error_recovery(state))
            state.update(await node_generate_answer(state))
            # 降级回答是非流式的，手动推送完整文本
            if state.get("final_answer"):
                yield _sse_event(
                    "token", {"text": state["final_answer"].answer}
                )
        else:
            yield _sse_event(
                "progress",
                {"node": "generate", "message": "正在生成答案..."},
            )

            # ── 生产者-消费者：real-time token push ──
            token_queue: asyncio.Queue = asyncio.Queue()

            async def on_token(t: str):
                """LLM 流式回调：每收到 token 立即入队"""
                await token_queue.put(("token", t))

            # 在后台任务中运行答案生成
            gen_task = asyncio.create_task(
                node_generate_answer(state, stream_callback=on_token)
            )

            # 边生成边消费：token 一入队就 yield
            has_tokens = False
            finished = False
            while not finished:
                # 等待 token 或任务完成
                try:
                    item = await asyncio.wait_for(token_queue.get(), timeout=0.05)
                    kind, payload = item
                    if kind == "token":
                        yield _sse_event("token", {"text": payload})
                        has_tokens = True
                        await asyncio.sleep(0)  # 让出事件循环给前端
                except asyncio.TimeoutError:
                    # 检查任务是否已完成
                    if gen_task.done():
                        finished = True

            # 确保任务完全结束
            await gen_task
            # 更新 state（gen_task 内部已经通过 stream_callback 推送了 token）
            state.update(gen_task.result())

            # 如果流式路径未产生任何 token（降级），发射完整答案
            if not has_tokens and state.get("final_answer"):
                yield _sse_event(
                    "token", {"text": state["final_answer"].answer}
                )

        # 如果答案生成也失败，二次降级
        if state.get("error") and state.get("final_answer") is None:
            state.update(await node_error_recovery(state))
            state.update(await node_generate_answer(state))
            if state.get("final_answer"):
                yield _sse_event(
                    "token", {"text": state["final_answer"].answer}
                )

        # ── 最终结果 ──
        final_answer: GeneratedAnswer = state["final_answer"]
        state["completed_at"] = datetime.now(timezone.utc).isoformat()

        # 来源
        sources = []
        if final_answer.sources:
            sources = [
                {"url": s.url, "title": s.title, "snippet": s.snippet}
                for s in final_answer.sources
            ]
        yield _sse_event("sources", {"sources": sources})

        # 完成
        yield _sse_event(
            "done",
            {
                "confidence": final_answer.confidence,
                "latency_ms": final_answer.latency_ms,
                "tokens_used": final_answer.tokens_used,
                "is_fallback": final_answer.is_fallback,
            },
        )

        # ── 更新历史 ──
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": final_answer.answer})
        max_turns = agent_cfg.max_history_turns
        if len(history) > max_turns * 2:
            agent._sessions[session_id] = history[-(max_turns * 2):]
        else:
            agent._sessions[session_id] = history

    except Exception as exc:
        yield _sse_event(
            "error",
            {"message": str(exc), "code": type(exc).__name__},
        )

    finally:
        # 恢复原始配置
        api_cfg.llm_model = orig_model
        agent_cfg.search_depth = orig_depth
        agent_cfg.top_k_fragments = orig_top_k
