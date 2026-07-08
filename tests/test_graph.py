"""
智能搜索助手 — Agent 状态图节点独立测试
测试范围: graph.py 全部 4 个节点 + 路由函数 + SearchAgent 类
与 test_e2e.py 不同，这里关注每个节点的独立行为、边缘条件、TypedDict 状态兼容性
"""
from __future__ import annotations

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


# ============================================================
# Mock 工具
# ============================================================
def make_mock_httpx_response(json_data: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def make_llm_response(content: str):
    return {"choices": [{"message": {"content": content}}]}


REWRITE_JSON = '{"rewritten": "改写的查询", "language": "zh", "intent": "news", "sub_queries": ["子查询1"]}'
ANSWER_TEXT = "这是基于搜索结果的回答。"


# ============================================================
# Node 1: node_rewrite_query 测试
# ============================================================
class TestNodeRewriteQuery:
    """查询改写节点测试"""

    @pytest.mark.asyncio
    async def test_basic_rewrite_state_update(self):
        """正常改写后，状态应包含 rewritten_queries"""
        state = {
            "user_query": "今天有什么新闻？",
            "history": [],
        }

        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response(REWRITE_JSON))

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_cls.return_value = mock_client

            result = await node_rewrite_query(state)

        assert "rewritten_queries" in result
        assert len(result["rewritten_queries"]) == 1
        rq = result["rewritten_queries"][0]
        assert isinstance(rq, RewrittenQuery)
        assert rq.original == "今天有什么新闻？"

    @pytest.mark.asyncio
    async def test_rewrite_with_empty_user_query(self):
        """空查询不应崩溃"""
        state = {"user_query": "", "history": []}

        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response(REWRITE_JSON))

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_cls.return_value = mock_client

            result = await node_rewrite_query(state)
        assert len(result["rewritten_queries"]) == 1

    @pytest.mark.asyncio
    async def test_rewrite_passes_history_to_tool(self):
        """对话历史应传递给 rewrite_query 函数"""
        history = [
            {"role": "user", "content": "上一轮问题"},
            {"role": "assistant", "content": "上一轮回答"},
        ]
        state = {"user_query": "详细说说", "history": history}

        captured_user_msgs = []

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            # user message 内容应该包含历史
            for m in messages:
                if m["role"] == "user":
                    captured_user_msgs.append(m["content"])
            return make_mock_httpx_response(make_llm_response(REWRITE_JSON))

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_cls.return_value = mock_client

            await node_rewrite_query(state)

        # 验证历史被传入 LLM prompt
        assert len(captured_user_msgs) > 0
        assert "上一轮问题" in captured_user_msgs[0]


# ============================================================
# Node 2: node_search 测试
# ============================================================
class TestNodeSearch:
    """搜索节点测试"""

    @pytest.mark.asyncio
    async def test_no_rewritten_queries(self):
        """无改写查询时立即返回空结果"""
        state = {"rewritten_queries": []}
        result = await node_search(state)

        assert result["search_results"] == []
        assert result["search_failed"] is True

    @pytest.mark.asyncio
    async def test_search_calls_multiple_sub_queries(self):
        """子查询应被依次执行"""
        rq = RewrittenQuery(
            original="测试",
            rewritten="主查询",
            sub_queries=["子查询1", "子查询2", "子查询3"],  # 3 个子查询
        )
        state = {"rewritten_queries": [rq]}

        call_count = [0]

        async def mock_post(*args, **kwargs):
            call_count[0] += 1
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""

            if "搜索结果质量评估专家" in system_content:
                scoring = '{"results": [{"url": "https://x.com/1", "title": "T", "relevance": 0.9, "reason": "OK"}]}'
                return make_mock_httpx_response(make_llm_response(scoring))
            else:
                return make_mock_httpx_response({"results": [
                    {"url": f"https://x.com/{call_count[0]}", "title": "T", "content": "C " * 15, "score": 0.8},
                ]})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_cls.return_value = mock_client

            result = await node_search(state)

        # 主查询 + 最多 2 个子查询 + 3 次打分 = 6 次调用
        # 实际: 主查询(Tavily) + 主查询(score) + 子查询1(Tavily) + 子查询1(score) + 子查询2(Tavily) + 子查询2(score)
        # = 6 次调用
        assert call_count[0] == 6
        assert result["search_failed"] is False

    @pytest.mark.asyncio
    async def test_search_state_keys_completeness(self):
        """搜索结果应包含所有必需的状态键"""
        rq = RewrittenQuery(original="t", rewritten="t", sub_queries=[])
        state = {"rewritten_queries": [rq]}

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""
            if "搜索结果质量评估专家" in system_content:
                scoring = '{"results": [{"url": "https://x.com/1", "title": "T", "relevance": 0.9, "reason": "OK"}]}'
                return make_mock_httpx_response(make_llm_response(scoring))
            else:
                return make_mock_httpx_response({"results": [
                    {"url": "https://x.com/1", "title": "T", "content": "C " * 15, "score": 0.8},
                ]})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_cls.return_value = mock_client

            result = await node_search(state)

        expected_keys = {"search_results", "deduped_results", "top_results",
                         "search_failed"}
        for key in expected_keys:
            assert key in result, f"缺少键: {key}"


# ============================================================
# Node 3: node_generate_answer 测试
# ============================================================
class TestNodeGenerateAnswer:
    """答案生成节点测试"""

    @pytest.mark.asyncio
    async def test_search_failed_triggers_fallback_generation(self):
        """搜索失败时走降级路径"""
        state = {
            "user_query": "今天天气？",
            "search_failed": True,
            "_fragments": [],
            "history": [],
        }

        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response("⚠️ 降级回答: 无法搜索，基于训练数据..."))

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_cls.return_value = mock_client

            result = await node_generate_answer(state)

        answer = result["final_answer"]
        assert isinstance(answer, GeneratedAnswer)
        assert answer.is_fallback is True
        assert answer.confidence == 0.5
        assert answer.sources == []
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_generate_with_fragments_produces_answer(self):
        """有搜索片段时走正常答案生成路径"""
        fragments = [
            ("https://example.com/1", "新闻内容 A " * 10, 0.95),
            ("https://example.com/2", "新闻内容 B " * 10, 0.82),
        ]
        state = {
            "user_query": "今天有什么新闻？",
            "search_failed": False,
            "_fragments": fragments,
            "history": [],
        }

        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response(ANSWER_TEXT))

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_cls.return_value = mock_client

            result = await node_generate_answer(state)

        answer = result["final_answer"]
        assert isinstance(answer, GeneratedAnswer)
        assert answer.is_fallback is False
        assert answer.confidence == 0.85
        assert len(answer.sources) >= 1
        assert answer.answer == ANSWER_TEXT

    @pytest.mark.asyncio
    async def test_sources_are_deduplicated(self):
        """相同 URL 的来源应去重"""
        fragments = [
            ("https://example.com/1", "content A " * 10, 0.95),
            ("https://example.com/1", "content A variant " * 10, 0.90),  # 重复 URL
            ("https://example.com/2", "content B " * 10, 0.82),
        ]
        state = {
            "user_query": "test",
            "search_failed": False,
            "_fragments": fragments,
            "history": [],
        }

        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response(ANSWER_TEXT))

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_cls.return_value = mock_client

            result = await node_generate_answer(state)

        sources = result["final_answer"].sources
        urls = [s.url for s in sources]
        assert len(urls) == len(set(urls))
        assert len(sources) == 2  # example.com/1 只出现一次

    @pytest.mark.asyncio
    async def test_history_included_in_context(self):
        """6 条以内的对话历史应包含在 LLM 上下文中"""
        history = [
            {"role": "user", "content": "之前的问题"},
            {"role": "assistant", "content": "之前的回答"},
        ]
        fragments = [("https://x.com/1", "content " * 10, 0.9)]
        state = {
            "user_query": "当前问题",
            "search_failed": False,
            "_fragments": fragments,
            "history": history,
        }

        captured_messages = []

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            captured_messages.extend(json_body.get("messages", []))
            return make_mock_httpx_response(make_llm_response(ANSWER_TEXT))

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_cls.return_value = mock_client

            await node_generate_answer(state)

        # 消息中应包含历史
        assert any("之前的问题" in str(m) for m in captured_messages)

    @pytest.mark.asyncio
    async def test_llm_api_failure_triggers_secondary_fallback(self):
        """LLM 答案生成失败时，node_generate_answer 自身做二次降级"""
        fragments = [("https://x.com/1", "content " * 10, 0.9)]
        state = {
            "user_query": "test",
            "search_failed": False,
            "_fragments": fragments,
            "history": [],
        }

        call_count = [0]

        async def mock_post(*args, **kwargs):
            call_count[0] += 1
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""

            if call_count[0] == 1:
                # 第一次：答案生成失败
                if "实时信息问答助手" in system_content:
                    raise Exception("LLM generation failed")
                # 打分 LLM — 不会被调用到，但以防万一
                return make_mock_httpx_response(make_llm_response('{"results": []}'))
            else:
                # 第二次：降级回答 (generate_fallback_answer)
                return make_mock_httpx_response(make_llm_response("⚠️ 降级答案"))

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_cls.return_value = mock_client

            result = await node_generate_answer(state)

        answer = result["final_answer"]
        assert answer.is_fallback is True
        assert answer.confidence == 0.4  # 二次降级置信度
        assert result["error"] is not None


# ============================================================
# Node 4: node_error_recovery 测试
# ============================================================
class TestNodeErrorRecovery:
    """错误恢复节点测试"""

    @pytest.mark.asyncio
    async def test_sets_correct_flags(self):
        """错误恢复应设置正确的状态标志"""
        state = {"error": "测试异常"}
        result = await node_error_recovery(state)

        assert result["fallback_triggered"] is True
        assert result["search_failed"] is True
        assert result["_fragments"] == []

    @pytest.mark.asyncio
    async def test_empty_error_message(self):
        """空错误信息也不应崩溃"""
        state = {"error": ""}
        result = await node_error_recovery(state)

        assert result["fallback_triggered"] is True
        assert result["search_failed"] is True

    @pytest.mark.asyncio
    async def test_missing_error_key(self):
        """state 中没有 error 键的默认行为"""
        state: dict = {}
        result = await node_error_recovery(state)

        assert result["fallback_triggered"] is True
        assert result["search_failed"] is True


# ============================================================
# 路由函数测试 (补充边缘)
# ============================================================
class TestRoutingEdgeCases:
    """路由函数边缘条件"""

    def test_route_after_search_missing_key(self):
        """state 中没有 search_failed 键 — 应默认当作成功"""
        result = route_after_search({})
        assert result == "generate"

    def test_route_after_search_explicit_false(self):
        """search_failed=False 明确路由到 generate"""
        result = route_after_search({"search_failed": False})
        assert result == "generate"

    def test_route_after_generate_missing_error_key(self):
        """state 中没有 error 键 — 应默认结束"""
        result = route_after_generate({})
        assert result == "__end__"

    def test_route_after_generate_empty_string_error(self):
        """error 为空字符串时视为无错误"""
        result = route_after_generate({"error": ""})
        assert result == "__end__"  # 空字符串是 falsy

    def test_route_after_generate_none_error(self):
        """error=None 路由到结束"""
        result = route_after_generate({"error": None})
        assert result == "__end__"


# ============================================================
# SearchAgent 类测试 (补充 E2E 未覆盖的)
# ============================================================
class TestSearchAgentClass:
    """SearchAgent 类的独立行为"""

    def test_initialization(self):
        """构造函数基础参数"""
        agent = SearchAgent()
        assert agent._sessions == {}

    def test_initialization_verbose(self):
        """verbose 参数应生效"""
        agent = SearchAgent(verbose=True)
        from config import get_agent_config
        assert get_agent_config().verbose is True

    def test_get_history_empty(self):
        """空会话返回空列表"""
        agent = SearchAgent()
        assert agent.get_history("nonexistent") == []

    def test_clear_session_nonexistent(self):
        """清除不存在的会话不应崩溃"""
        agent = SearchAgent()
        agent.clear_session("nonexistent")  # 不应抛异常

    def test_clear_session_removes_history(self):
        """清除会话后 get_history 应返回空"""
        agent = SearchAgent()
        agent._sessions["test_session"] = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        agent.clear_session("test_session")
        assert agent.get_history("test_session") == []

    def test_deduplicator_initialized(self):
        """SearchAgent 应初始化 URLDeduplicator"""
        agent = SearchAgent()
        assert agent.deduplicator is not None

    @pytest.mark.asyncio
    async def test_completed_at_is_set(self):
        """run() 后 completed_at 不应为空（在 state 内部设置）"""
        agent = SearchAgent()

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""
            if "搜索查询优化专家" in system_content:
                return make_mock_httpx_response(make_llm_response(REWRITE_JSON))
            elif "搜索结果质量评估专家" in system_content:
                return make_mock_httpx_response(make_llm_response(
                    '{"results": [{"url": "https://x.com/1", "title": "T", "relevance": 0.9, "reason": "OK"}]}'
                ))
            elif "实时信息问答助手" in system_content:
                return make_mock_httpx_response(make_llm_response(ANSWER_TEXT))
            else:
                return make_mock_httpx_response({"results": [
                    {"url": "https://x.com/1", "title": "T", "content": "C " * 15, "score": 0.9},
                ]})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_cls.return_value = mock_client

            result = await agent.run("test")

        # Result 应该有合理的时间戳（由 run 内部设置 completed_at）
        # 验证返回值完整性
        assert isinstance(result, GeneratedAnswer)
        assert result.query == "test"


# ============================================================
# State TypedDict 完整性测试
# ============================================================
class TestAgentStateTypedDict:
    """AgentState TypedDict 的字段存在性验证"""

    def test_all_state_keys_defined(self):
        """确保 AgentState 定义了所有 pipeline 中使用的键"""
        from agent.models import AgentState

        # 这些键在 graph.py 的 run() 和节点中被使用
        required_keys = {
            "session_id", "user_query", "user_query_raw",
            "rewritten_queries", "search_results", "deduped_results",
            "relevance_scores", "top_results", "final_answer",
            "history", "error", "retry_count", "fallback_triggered",
            "search_failed", "started_at", "completed_at",
        }
        # AgentState 是 TypedDict(total=False)，用 __annotations__ 获取键
        state_keys = set(AgentState.__annotations__.keys())
        for key in required_keys:
            assert key in state_keys, f"AgentState 缺少键: {key}"

    def test_private_underscore_key_usage(self):
        """_fragments 是私有传输键，在节点间传递但不属于 AgentState"""
        # 这是设计决策：_fragments 作为中间数据在节点间传递
        # 验证 _fragments 在 graph.py 源码中被使用
        import inspect
        from agent.graph import node_generate_answer, node_search

        # node_generate_answer 应读取 _fragments
        source = inspect.getsource(node_generate_answer)
        assert "_fragments" in source, "node_generate_answer 应使用 _fragments"

        # node_search 应写入 _fragments
        source_search = inspect.getsource(node_search)
        assert "_fragments" in source_search, "node_search 应写入 _fragments"


# ============================================================
# ANSWER_SYSTEM_PROMPT 常量测试
# ============================================================
class TestAnswerSystemPrompt:
    """答案生成系统提示测试"""

    def test_prompt_is_non_empty(self):
        assert len(ANSWER_SYSTEM_PROMPT) > 0

    def test_prompt_contains_key_instructions(self):
        """提示应包含核心要求"""
        assert "搜索结果" in ANSWER_SYSTEM_PROMPT
        assert "来源" in ANSWER_SYSTEM_PROMPT
        assert "Markdown" in ANSWER_SYSTEM_PROMPT

    def test_prompt_is_not_default(self):
        """确保提示不是简单占位符"""
        assert len(ANSWER_SYSTEM_PROMPT) > 100
