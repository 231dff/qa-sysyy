"""
智能搜索助手 — LangGraph Agent 状态图
核心工作流: 查询改写 → 实时搜索 → 结果过滤 → 答案生成
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Literal, Optional

import httpx

from agent.models import (
    AgentState,
    GeneratedAnswer,
    RewrittenQuery,
    Source,
)
from config import get_agent_config, get_api_config
from tools.registry import (
    generate_fallback_answer,
    rewrite_query,
    search_and_filter_pipeline,
)
from utils.helpers import (
    URLDeduplicator,
    count_tokens,
    format_search_context,
    generate_session_id,
)


# ============================================================
# 答案生成系统提示
# ============================================================
ANSWER_SYSTEM_PROMPT = """你是一个实时信息问答助手。你将获得用户的原始问题以及从互联网实时搜索到的相关信息。

## 回答规则
1. **基于搜索信息回答**: 优先使用提供的搜索结果，而非你的训练数据
2. **标注来源**: 每一条关键信息标注来源编号 [1] [2] 等
3. **诚实**: 如果搜索结果不足以回答，明确说明，不要编造
4. **结构化**: 使用 Markdown 格式，适当使用标题、列表、引用
5. **时效提示**: 如果问题涉及实时信息，注明搜索时间
6. **简洁准确**: 直接回答核心问题，避免冗长铺垫

## 输出格式 (Markdown)
- 开头直接回答问题
- 中间展开细节，标注来源
- 结尾列出「📚 参考来源」"""


# ============================================================
# LangGraph Node 1: 查询改写
# ============================================================
async def node_rewrite_query(state: AgentState) -> dict:
    """
    节点: 查询改写
    - 将用户口语化输入改写为搜索引擎友好查询
    - 结合对话历史理解多轮意图
    """
    agent_cfg = get_agent_config()

    user_query = state.get("user_query", "")
    history = state.get("history", [])

    if agent_cfg.verbose:
        print(f"[节点: 查询改写] 原始: {user_query[:100]}...")

    rewritten = await rewrite_query(user_query, history)

    if agent_cfg.verbose:
        print(f"  改写后: {rewritten.rewritten}")
        if rewritten.sub_queries:
            print(f"  子查询: {rewritten.sub_queries}")

    return {
        "rewritten_queries": [rewritten],
    }


# ============================================================
# LangGraph Node 2: 实时搜索
# ============================================================
async def node_search(state: AgentState) -> dict:
    """
    节点: 实时搜索
    - 使用改写后的查询及子查询执行 Tavily 搜索
    - URL 去重 + LLM 相关性打分 + Token 裁剪
    """
    agent_cfg = get_agent_config()

    rewritten_list = state.get("rewritten_queries", [])
    if not rewritten_list:
        return {"search_results": [], "search_failed": True}

    rewritten = rewritten_list[0]

    # 搜集所有查询词
    queries = [rewritten.rewritten]
    queries.extend(rewritten.sub_queries[:2])  # 最多 2 个子查询

    all_deduped: list = []
    all_fragments: list = []
    deduplicator = URLDeduplicator(window_size=agent_cfg.dedup_window)

    for q in queries:
        if agent_cfg.verbose:
            print(f"[节点: 实时搜索] 搜索: {q}")

        deduped_results, fragments = await search_and_filter_pipeline(
            q, deduplicator=deduplicator
        )
        all_deduped.extend(deduped_results)
        all_fragments.extend(fragments)

    search_failed = len(all_deduped) == 0

    if agent_cfg.verbose:
        print(f"[节点: 实时搜索] 有效结果: {len(all_deduped)} 条, "
              f"裁剪后片段: {len(all_fragments)} 条, 搜索失败: {search_failed}")

    return {
        "search_results": all_deduped,
        "deduped_results": all_deduped,
        "top_results": all_deduped,
        "_fragments": all_fragments,  # 私有字段，传给下一个节点
        "search_failed": search_failed,
    }


# ============================================================
# LangGraph Node 3: 生成答案
# ============================================================
async def node_generate_answer(state: AgentState) -> dict:
    """
    节点: 生成答案
    - 如果搜索成功：基于搜索片段生成引用答案
    - 如果搜索失败：降级为 LLM 参数化知识回答
    """
    agent_cfg = get_agent_config()
    api_cfg = get_api_config()

    user_query = state.get("user_query", "")
    search_failed = state.get("search_failed", False)
    fragments = state.get("_fragments", [])
    history = state.get("history", [])

    start = time.perf_counter()

    # ── 搜索失败 → 降级回答 ──
    if search_failed or not fragments:
        if agent_cfg.verbose:
            print("[节点: 生成答案] 搜索失败, 使用降级回答")

        fallback_text = await generate_fallback_answer(user_query, history)
        elapsed_ms = (time.perf_counter() - start) * 1000

        answer = GeneratedAnswer(
            query=user_query,
            answer=fallback_text,
            sources=[],
            confidence=0.5,
            is_fallback=True,
            tokens_used=count_tokens(fallback_text),
            latency_ms=elapsed_ms,
        )
        return {"final_answer": answer, "error": None}

    # ── 搜索成功 → 基于上下文的引用答案 ──
    context_text = format_search_context(fragments)

    user_message = f"""## 用户问题
{user_query}

## 实时搜索结果 (当前时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})
{context_text}

请基于以上搜索结果回答问题:"""

    messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
    if history:
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"][:500]})
    messages.append({"role": "user", "content": user_message})

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{api_cfg.llm_api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_cfg.llm_api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": api_cfg.llm_model,
                    "temperature": 0.3,
                    "messages": messages,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            answer_text = data["choices"][0]["message"]["content"]

    except Exception as exc:
        if agent_cfg.verbose:
            print(f"[答案生成失败] {exc}")
        # 二次降级
        fallback_text = await generate_fallback_answer(user_query, history)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "final_answer": GeneratedAnswer(
                query=user_query,
                answer=fallback_text,
                sources=[],
                confidence=0.4,
                is_fallback=True,
                tokens_used=count_tokens(fallback_text),
                latency_ms=elapsed_ms,
            ),
            "error": str(exc),
        }

    elapsed_ms = (time.perf_counter() - start) * 1000

    # 构建来源列表
    sources_list = []
    seen_urls = set()
    for url, content, score in fragments:
        if url not in seen_urls:
            sources_list.append(Source(
                url=url,
                title="",
                snippet=content[:200],
            ))
            seen_urls.add(url)

    answer = GeneratedAnswer(
        query=user_query,
        answer=answer_text,
        sources=sources_list,
        confidence=0.85,
        is_fallback=False,
        tokens_used=count_tokens(answer_text),
        latency_ms=elapsed_ms,
    )

    return {"final_answer": answer, "error": None}


# ============================================================
# LangGraph Node 4: 错误恢复
# ============================================================
async def node_error_recovery(state: AgentState) -> dict:
    """
    节点: 错误恢复
    - 捕获异常，触发降级回答
    """
    error_msg = state.get("error", "未知错误")
    if get_agent_config().verbose:
        print(f"[节点: 错误恢复] {error_msg}")

    return {
        "fallback_triggered": True,
        "search_failed": True,
        "_fragments": [],
    }


# ============================================================
# 路由函数
# ============================================================
def route_after_search(state: AgentState) -> Literal["generate", "error"]:
    """
    搜索后路由:
    - 搜索成功 → 生成答案
    - 搜索失败 → 错误恢复 (触发降级回答)
    """
    if state.get("search_failed", False):
        return "error"
    return "generate"


def route_after_generate(state: AgentState) -> Literal["__end__", "error"]:
    """
    答案生成后路由:
    - 成功 → 结束
    - 失败 → 错误恢复
    """
    if state.get("error"):
        return "error"
    return "__end__"


# ============================================================
# Agent 主类
# ============================================================
class SearchAgent:
    """
    智能搜索 Agent

    用法:
        agent = SearchAgent()
        answer = await agent.run("今天有什么重大新闻？")
        print(answer.answer)
    """

    def __init__(self, verbose: bool = False):
        if verbose:
            get_agent_config().verbose = True

        self.deduplicator = URLDeduplicator(
            window_size=get_agent_config().dedup_window
        )
        self._sessions: dict[str, list[dict[str, str]]] = {}

    async def run(
        self,
        user_query: str,
        session_id: Optional[str] = None,
    ) -> GeneratedAnswer:
        """
        执行完整的搜索-回答流水线

        Args:
            user_query: 用户问题
            session_id: 会话 ID (None 则自动生成)

        Returns:
            GeneratedAnswer 对象
        """
        agent_cfg = get_agent_config()

        # 会话管理
        if session_id is None:
            session_id = generate_session_id()
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        history = self._sessions[session_id]

        if agent_cfg.verbose:
            print(f"\n{'='*60}")
            print(f"[Agent] session={session_id}, query={user_query[:80]}...")
            print(f"[Agent] 历史轮次: {len(history)//2}")

        # ── 构建初始状态 ──
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

        # ── 执行流水线 (显式调用，无需 LangGraph 编译) ──
        # Step 1: 查询改写
        state.update(await node_rewrite_query(state))

        # Step 2: 实时搜索
        state.update(await node_search(state))

        # Step 3: 路由 → 生成答案 或 错误恢复
        route = route_after_search(state)
        if route == "error":
            state.update(await node_error_recovery(state))
            state.update(await node_generate_answer(state))
        else:
            state.update(await node_generate_answer(state))

        # 如果生成阶段也失败了
        if state.get("error") and state.get("final_answer") is None:
            state.update(await node_error_recovery(state))
            state.update(await node_generate_answer(state))

        state["completed_at"] = datetime.now(timezone.utc).isoformat()

        # ── 更新对话历史 ──
        history.append({"role": "user", "content": user_query})
        answer_text = state["final_answer"].answer
        history.append({"role": "assistant", "content": answer_text})

        # 裁剪历史
        max_turns = agent_cfg.max_history_turns
        if len(history) > max_turns * 2:
            self._sessions[session_id] = history[-(max_turns * 2):]
        else:
            self._sessions[session_id] = history

        # ── 清理私有字段 ──
        final_answer = state["final_answer"]

        if agent_cfg.verbose:
            status = "⚠️ 降级" if final_answer.is_fallback else "✅ 正常"
            print(f"[Agent] 完成 {status} | 置信度:{final_answer.confidence:.0%} "
                  f"| 耗时:{final_answer.latency_ms:.0f}ms "
                  f"| Token:{final_answer.tokens_used}")
            print(f"{'='*60}\n")

        return final_answer

    async def run_stream(self, user_query: str, session_id: str = "default"):
        """
        流式运行 (用于 Streamlit 前端)
        目前通过 yield 分步输出状态
        """
        yield {"type": "status", "message": "🔄 正在改写查询..."}
        result = await self.run(user_query, session_id)
        yield {"type": "status", "message": "✅ 完成"}
        yield {"type": "answer", "data": result}

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """获取会话历史"""
        return self._sessions.get(session_id, [])

    def clear_session(self, session_id: str) -> None:
        """清除会话"""
        self._sessions.pop(session_id, None)
        self.deduplicator.clear()
