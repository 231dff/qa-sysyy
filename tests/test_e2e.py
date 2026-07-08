"""
智能搜索助手 — Agent 完整流水线端到端测试
测试范围: 查询改写→搜索→路由→答案生成→错误恢复→历史更新
覆盖 graph.py#L366-L384 的完整链路

使用 mock 外部 API 调用 (httpx.AsyncClient) 以避免实际网络依赖。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.graph import (
    ANSWER_SYSTEM_PROMPT,
    SearchAgent,
    node_error_recovery,
    node_generate_answer,
    node_rewrite_query,
    node_search,
    route_after_generate,
    route_after_search,
)
from agent.models import GeneratedAnswer, RewrittenQuery, Source
from config import AgentConfig, get_agent_config


# ============================================================
# Mock 工具函数
# ============================================================
def make_mock_httpx_response(json_data: dict, status_code: int = 200):
    """构造 mock httpx.Response"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def make_llm_response(content: str):
    """构造标准 LLM API 响应"""
    return {
        "choices": [
            {"message": {"content": content}}
        ]
    }


def make_tavily_response(results: list[dict] | None = None):
    """构造标准 Tavily API 响应"""
    if results is None:
        results = [
            {
                "url": "https://example.com/news/1",
                "title": "测试新闻标题",
                "content": "这是测试新闻内容，包含一些关键信息。",
                "score": 0.95,
            },
            {
                "url": "https://example.com/news/2",
                "title": "第二篇测试新闻",
                "content": "另一篇相关的测试内容。",
                "score": 0.82,
            },
        ]
    return {"results": results}


REWRITE_LLM_OUTPUT = """{
  "rewritten": "2025年7月 重大新闻",
  "language": "zh",
  "intent": "news",
  "sub_queries": ["中国新闻 2025年7月", "国际新闻 2025年7月"]
}"""

FALLBACK_LLM_OUTPUT = "⚠️ 当前无法进行实时搜索。根据我的训练数据，这是一条测试降级回答。"

ANSWER_LLM_OUTPUT = """## 今日要闻

根据最新搜索结果显示，今天有以下重要新闻：

1. **AI 技术突破** [1]: 某公司发布了最新的多模态 AI 模型
2. **科技行业动态** [2]: 多家科技巨头发布了季度财报

📚 参考来源"""


# ============================================================
# 端到端: 查询改写 → 搜索 → 路由 → 答案生成
# ============================================================
class TestAgentPipelineE2E:
    """Agent 完整流水线端到端测试"""

    # ── 正常路径 (Happy Path) ──

    @pytest.mark.asyncio
    async def test_full_pipeline_happy_path(self):
        """完整的正常流程: 查询改写→搜索→答案生成, 验证最终答案"""
        agent = SearchAgent()
        query = "今天有什么重大新闻？"

        # Mock httpx.AsyncClient.post — 按调用顺序返回:
        # 第1次: 查询改写 LLM
        # 第2次: Tavily 搜索 (query="2025年7月 重大新闻")
        # 第3次: Tavily 搜索 (query="中国新闻 2025年7月") — 子查询1
        # 第4次: Tavily 搜索 (query="国际新闻 2025年7月") — 子查询2
        # 第5次: 相关性打分 LLM (search_and_filter_pipeline → score_relevance)
        # 第6次: 相关性打分 LLM (第2个子查询)
        # 第7次: 相关性打分 LLM (第3个子查询)
        # 第8次: 答案生成 LLM

        call_count = [0]

        async def mock_post(*args, **kwargs):
            call_count[0] += 1
            # 根据请求体内容判断是哪个调用
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""

            if "搜索查询优化专家" in system_content:
                # 查询改写 LLM
                return make_mock_httpx_response(make_llm_response(REWRITE_LLM_OUTPUT))
            elif "搜索结果质量评估专家" in system_content:
                # 相关性打分 LLM — 返回高分
                scoring_output = """{
                  "results": [
                    {"url": "https://example.com/news/1", "title": "测试新闻标题", "relevance": 0.95, "reason": "直接相关"},
                    {"url": "https://example.com/news/2", "title": "第二篇测试新闻", "relevance": 0.82, "reason": "高度相关"}
                  ]
                }"""
                return make_mock_httpx_response(make_llm_response(scoring_output))
            elif "实时信息问答助手" in system_content or "ANSWER_SYSTEM_PROMPT" in str(messages):
                # 答案生成 LLM
                return make_mock_httpx_response(make_llm_response(ANSWER_LLM_OUTPUT))
            else:
                # Tavily 搜索 API
                return make_mock_httpx_response(make_tavily_response())

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await agent.run(query)

        # 验证结果
        assert isinstance(result, GeneratedAnswer)
        assert result.query == query
        assert len(result.answer) > 0
        assert result.is_fallback is False
        assert result.confidence == 0.85
        assert result.latency_ms > 0
        assert result.tokens_used >= 0

        # 验证来源
        assert len(result.sources) >= 1
        for src in result.sources:
            assert isinstance(src, Source)
            assert src.url.startswith("https://")

        # 验证历史更新
        all_sessions = list(agent._sessions.keys())
        assert len(all_sessions) == 1
        session_id = all_sessions[0]
        history = agent.get_history(session_id)
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[0]["content"] == query
        assert history[1]["role"] == "assistant"

    # ── 搜索失败 → 降级回答 ──

    @pytest.mark.asyncio
    async def test_search_failure_triggers_fallback(self):
        """搜索失败时, 应触发降级回答路径 (error recovery → fallback answer)"""
        agent = SearchAgent()
        query = "今天有什么重大新闻？"

        call_count = [0]

        async def mock_post(*args, **kwargs):
            call_count[0] += 1
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""

            if "搜索查询优化专家" in system_content:
                # 查询改写正常
                return make_mock_httpx_response(make_llm_response(REWRITE_LLM_OUTPUT))
            elif "搜索结果质量评估专家" in system_content:
                # 不会走到这里 (搜索已经失败)
                return make_mock_httpx_response(make_llm_response("{}"))
            elif "实时信息问答助手" in system_content or "智能问答助手" in system_content:
                # 降级回答或正常答案生成
                return make_mock_httpx_response(make_llm_response(FALLBACK_LLM_OUTPUT))
            else:
                # Tavily 搜索 — 返回空结果 (模拟搜索失败)
                return make_mock_httpx_response({"results": []})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await agent.run(query)

        # 验证降级回答
        assert isinstance(result, GeneratedAnswer)
        assert result.is_fallback is True
        assert result.confidence == 0.5  # 降级回答的默认置信度
        assert len(result.sources) == 0
        assert len(result.answer) > 0

    # ── 答案生成失败 → 二次降级 ──

    @pytest.mark.asyncio
    async def test_answer_generation_failure_triggers_secondary_fallback(self):
        """答案生成失败时, 应进入二次降级 (LLM API 异常 → 再次调用 fallback)"""
        agent = SearchAgent()
        query = "今天有什么重大新闻？"

        call_count = [0]

        async def mock_post(*args, **kwargs):
            call_count[0] += 1
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""

            if "搜索查询优化专家" in system_content:
                return make_mock_httpx_response(make_llm_response(REWRITE_LLM_OUTPUT))
            elif "搜索结果质量评估专家" in system_content:
                scoring_output = """{
                  "results": [
                    {"url": "https://example.com/news/1", "title": "Test", "relevance": 0.9, "reason": "相关"}
                  ]
                }"""
                return make_mock_httpx_response(make_llm_response(scoring_output))
            elif "实时信息问答助手" in system_content:
                # 模拟答案生成 LLM 失败 (抛异常)
                raise Exception("LLM API 500 Internal Server Error")
            elif "智能问答助手" in system_content:
                # 二次降级: fallback answer generator
                return make_mock_httpx_response(make_llm_response(FALLBACK_LLM_OUTPUT))
            else:
                # Tavily 搜索正常
                return make_mock_httpx_response(make_tavily_response())

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await agent.run(query)

        # 验证二次降级
        assert isinstance(result, GeneratedAnswer)
        assert result.is_fallback is True
        assert result.confidence == 0.4  # 二次降级的置信度更低
        assert len(result.answer) > 0

    # ── 多轮对话 ──

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_history(self):
        """多轮对话: 历史应正确累积, 并在后续查询中传递"""
        agent = SearchAgent()
        session_id = "test-multi-turn"

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""

            if "搜索查询优化专家" in system_content:
                return make_mock_httpx_response(make_llm_response(REWRITE_LLM_OUTPUT))
            elif "搜索结果质量评估专家" in system_content:
                return make_mock_httpx_response(make_llm_response("""{
                  "results": [
                    {"url": "https://example.com/1", "title": "T", "relevance": 0.9, "reason": "相关"}
                  ]
                }"""))
            elif "实时信息问答助手" in system_content:
                return make_mock_httpx_response(make_llm_response(ANSWER_LLM_OUTPUT))
            else:
                return make_mock_httpx_response(make_tavily_response())

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            # 第一轮
            result1 = await agent.run("今天有什么新闻？", session_id=session_id)
            history1 = agent.get_history(session_id)
            assert len(history1) == 2

            # 第二轮
            result2 = await agent.run("详细说说第一条", session_id=session_id)
            history2 = agent.get_history(session_id)
            assert len(history2) == 4  # 两轮对话, 4 条消息

            # 历史顺序正确
            assert history2[0]["role"] == "user"
            assert history2[0]["content"] == "今天有什么新闻？"
            assert history2[1]["role"] == "assistant"
            assert history2[2]["role"] == "user"
            assert history2[2]["content"] == "详细说说第一条"
            assert history2[3]["role"] == "assistant"

    # ── 历史裁剪 ──

    @pytest.mark.asyncio
    async def test_history_trimming(self):
        """历史超过 max_history_turns 时应被裁剪"""
        agent = SearchAgent()
        session_id = "test-trim"

        # 临时修改配置以加速测试
        agent_cfg = get_agent_config()
        original_max_turns = agent_cfg.max_history_turns
        agent_cfg.max_history_turns = 2  # 只保留 2 轮

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""

            if "搜索查询优化专家" in system_content:
                return make_mock_httpx_response(make_llm_response(REWRITE_LLM_OUTPUT))
            elif "搜索结果质量评估专家" in system_content:
                return make_mock_httpx_response(make_llm_response("""{
                  "results": [{"url": "https://x.com/1", "title": "T", "relevance": 0.9, "reason": "OK"}]
                }"""))
            elif "实时信息问答助手" in system_content:
                return make_mock_httpx_response(make_llm_response(ANSWER_LLM_OUTPUT))
            else:
                return make_mock_httpx_response(make_tavily_response())

        try:
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.post = AsyncMock(side_effect=mock_post)
                mock_client_cls.return_value = mock_client

                # 发送 4 轮对话 (超过 max_history_turns=2)
                for i in range(4):
                    result = await agent.run(f"问题 {i+1}", session_id=session_id)

                history = agent.get_history(session_id)
                # 最多 2 轮 = 4 条消息
                assert len(history) <= agent_cfg.max_history_turns * 2
                assert len(history) == 4
                # 应该只保留最后 2 轮 (问题3→回答3→问题4→回答4)
                assert history[0]["content"] == "问题 3"
                assert history[2]["content"] == "问题 4"

        finally:
            agent_cfg.max_history_turns = original_max_turns

    # ── 查询改写失败 → 回退到原始查询 ──

    @pytest.mark.asyncio
    async def test_rewrite_failure_falls_back_to_original_query(self):
        """查询改写 LLM 异常时, 使用原始用户输入继续流程"""
        agent = SearchAgent()
        query = "今天有什么重大新闻？"

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""

            if "搜索查询优化专家" in system_content:
                # 查询改写失败
                raise Exception("LLM rewrite service unavailable")
            elif "搜索结果质量评估专家" in system_content:
                return make_mock_httpx_response(make_llm_response("""{
                  "results": [{"url": "https://x.com/1", "title": "T", "relevance": 0.5, "reason": "一般"}]
                }"""))
            elif "实时信息问答助手" in system_content:
                return make_mock_httpx_response(make_llm_response(ANSWER_LLM_OUTPUT))
            else:
                return make_mock_httpx_response(make_tavily_response())

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await agent.run(query)

        # 即使改写失败, 也应该成功返回结果
        assert isinstance(result, GeneratedAnswer)
        assert result.query == query
        assert len(result.answer) > 0


# ============================================================
# 路由函数测试
# ============================================================
class TestRoutingLogic:
    """路由逻辑独立测试"""

    def test_route_after_search_success(self):
        """搜索成功时路由到 generate"""
        state = {"search_failed": False}
        assert route_after_search(state) == "generate"

    def test_route_after_search_failure(self):
        """搜索失败时路由到 error"""
        state = {"search_failed": True}
        assert route_after_search(state) == "error"

    def test_route_after_generate_success(self):
        """生成成功时路由到结束"""
        state = {"error": None}
        assert route_after_generate(state) == "__end__"

    def test_route_after_generate_failure(self):
        """生成失败时路由到 error 恢复"""
        state = {"error": "something went wrong"}
        assert route_after_generate(state) == "error"


# ============================================================
# 单节点行为测试 (辅助)
# ============================================================
class TestNodeBehaviors:
    """单个节点独立行为测试"""

    # ── 错误恢复节点 ──

    @pytest.mark.asyncio
    async def test_error_recovery_node(self):
        """错误恢复节点应设置正确的状态标志"""
        state = {"error": "测试错误信息"}
        result = await node_error_recovery(state)
        assert result["fallback_triggered"] is True
        assert result["search_failed"] is True
        assert result["_fragments"] == []

    # ── 搜索节点 (无改写查询) ──

    @pytest.mark.asyncio
    async def test_search_node_without_rewritten_queries(self):
        """没有改写查询时, 搜索节点应立即返回失败"""
        state = {"rewritten_queries": []}
        result = await node_search(state)
        assert result["search_results"] == []
        assert result["search_failed"] is True

    # ── 答案生成节点 (搜索失败) ──

    @pytest.mark.asyncio
    async def test_generate_node_search_failed(self):
        """搜索失败时, 答案生成节点应走降级路径"""
        state = {
            "user_query": "test",
            "search_failed": True,
            "_fragments": [],
            "history": [],
        }

        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response(FALLBACK_LLM_OUTPUT))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await node_generate_answer(state)

        assert result["final_answer"].is_fallback is True
        assert result["final_answer"].confidence == 0.5
        assert result["error"] is None


# ============================================================
# Session 管理 & 清理测试
# ============================================================
class TestSessionManagement:
    """会话管理相关测试"""

    @pytest.mark.asyncio
    async def test_auto_session_id_generation(self):
        """未提供 session_id 时自动生成"""
        agent = SearchAgent()

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""
            if "搜索查询优化专家" in system_content:
                return make_mock_httpx_response(make_llm_response(REWRITE_LLM_OUTPUT))
            elif "搜索结果质量评估专家" in system_content:
                return make_mock_httpx_response(make_llm_response("""{
                  "results": [{"url": "https://x.com/1", "title": "T", "relevance": 0.8, "reason": "OK"}]
                }"""))
            elif "实时信息问答助手" in system_content:
                return make_mock_httpx_response(make_llm_response("答案"))
            else:
                return make_mock_httpx_response(make_tavily_response())

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await agent.run("test")

        # 应该有 1 个自动生成的 session
        assert len(agent._sessions) == 1

    @pytest.mark.asyncio
    async def test_clear_session(self):
        """清除会话后历史为空"""
        agent = SearchAgent()
        session_id = "test-clear"

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""
            if "搜索查询优化专家" in system_content:
                return make_mock_httpx_response(make_llm_response(REWRITE_LLM_OUTPUT))
            elif "搜索结果质量评估专家" in system_content:
                return make_mock_httpx_response(make_llm_response("""{
                  "results": [{"url": "https://x.com/1", "title": "T", "relevance": 0.9, "reason": "OK"}]
                }"""))
            elif "实时信息问答助手" in system_content:
                return make_mock_httpx_response(make_llm_response("答案"))
            else:
                return make_mock_httpx_response(make_tavily_response())

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            await agent.run("问题1", session_id=session_id)
            assert len(agent.get_history(session_id)) == 2

            agent.clear_session(session_id)
            assert agent.get_history(session_id) == []


# ============================================================
# 集成: 结果验证
# ============================================================
class TestResultValidation:
    """端到端结果的完整性验证"""

    @pytest.mark.asyncio
    async def test_answer_contains_sources_when_search_succeeds(self):
        """搜索成功时, 答案应包含来源"""
        agent = SearchAgent()

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""
            if "搜索查询优化专家" in system_content:
                return make_mock_httpx_response(make_llm_response(REWRITE_LLM_OUTPUT))
            elif "搜索结果质量评估专家" in system_content:
                return make_mock_httpx_response(make_llm_response("""{
                  "results": [
                    {"url": "https://example.com/a", "title": "A", "relevance": 0.95, "reason": "直接相关"},
                    {"url": "https://example.com/b", "title": "B", "relevance": 0.88, "reason": "高度相关"}
                  ]
                }"""))
            elif "实时信息问答助手" in system_content:
                return make_mock_httpx_response(make_llm_response(ANSWER_LLM_OUTPUT))
            else:
                return make_mock_httpx_response(make_tavily_response())

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await agent.run("test query")

        assert len(result.sources) > 0
        # 来源应该已去重
        urls = [s.url for s in result.sources]
        assert len(urls) == len(set(urls))

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self):
        """每次运行应记录延迟"""
        agent = SearchAgent()

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""
            if "搜索查询优化专家" in system_content:
                return make_mock_httpx_response(make_llm_response(REWRITE_LLM_OUTPUT))
            elif "搜索结果质量评估专家" in system_content:
                return make_mock_httpx_response(make_llm_response("""{
                  "results": [{"url": "https://x.com/1", "title": "T", "relevance": 0.9, "reason": "OK"}]
                }"""))
            elif "实时信息问答助手" in system_content:
                return make_mock_httpx_response(make_llm_response("答案"))
            else:
                return make_mock_httpx_response(make_tavily_response())

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await agent.run("test")

        assert result.latency_ms > 0
        assert result.tokens_used >= 0

    @pytest.mark.asyncio
    async def test_stream_yields_status_and_answer(self):
        """流式输出应依次产出状态和答案"""
        agent = SearchAgent()

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""
            if "搜索查询优化专家" in system_content:
                return make_mock_httpx_response(make_llm_response(REWRITE_LLM_OUTPUT))
            elif "搜索结果质量评估专家" in system_content:
                return make_mock_httpx_response(make_llm_response("""{
                  "results": [{"url": "https://x.com/1", "title": "T", "relevance": 0.9, "reason": "OK"}]
                }"""))
            elif "实时信息问答助手" in system_content:
                return make_mock_httpx_response(make_llm_response("答案"))
            else:
                return make_mock_httpx_response(make_tavily_response())

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            outputs = []
            async for item in agent.run_stream("test query"):
                outputs.append(item)

        assert len(outputs) >= 2
        assert outputs[0]["type"] == "status"
        assert outputs[-1]["type"] == "answer"
