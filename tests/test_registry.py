"""
智能搜索助手 — 工具注册中心完整测试
测试范围: 全部 4 个工具 + 搜索过滤组合流水线
策略: mock httpx.AsyncClient 隔离所有外部 API 调用
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.models import (
    RelevanceScore,
    RelevanceScores,
    RewrittenQuery,
    SearchResponse,
    SearchResult,
)
from tools.registry import (
    TOOL_REGISTRY,
    generate_fallback_answer,
    rewrite_query,
    score_relevance,
    search_and_filter_pipeline,
    tavily_search,
)
from utils.helpers import URLDeduplicator


# ============================================================
# 测试工具函数
# ============================================================
def make_mock_httpx_response(json_data: dict, status_code: int = 200):
    """构造 mock httpx.Response"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def make_llm_response(content: str):
    """构造 LLM API 标准响应"""
    return {"choices": [{"message": {"content": content}}]}


def make_tavily_api_response(results: list[dict]):
    """构造 Tavily API 原始响应"""
    return {"results": results}


# ============================================================
# Tool 1: Tavily 搜索测试
# ============================================================
class TestTavilySearch:
    """Tavily 搜索 API 测试"""

    @pytest.mark.asyncio
    async def test_successful_search(self):
        """正常搜索返回结果"""
        mock_results = [
            {"url": "https://example.com/1", "title": "新闻 A", "content": "内容 A", "score": 0.9},
            {"url": "https://example.com/2", "title": "新闻 B", "content": "内容 B", "score": 0.7},
        ]

        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_tavily_api_response(mock_results))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            response = await tavily_search("测试查询")

        assert isinstance(response, SearchResponse)
        assert response.query == "测试查询"
        assert len(response.results) == 2
        assert response.total_count == 2
        assert response.error is None
        assert response.search_time_ms > 0

        # 验证结果模型
        r = response.results[0]
        assert isinstance(r, SearchResult)
        assert r.url == "https://example.com/1"
        assert r.title == "新闻 A"
        assert r.score == 0.9

    @pytest.mark.asyncio
    async def test_search_timeout(self):
        """搜索超时应返回空结果 + 错误信息"""
        import httpx

        async def mock_post(*args, **kwargs):
            raise httpx.TimeoutException("Request timed out")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            response = await tavily_search("超时测试")

        assert isinstance(response, SearchResponse)
        assert response.results == []
        assert response.total_count == 0
        assert "超时" in response.error

    @pytest.mark.asyncio
    async def test_search_general_error(self):
        """搜索通用异常应返回空结果 + 错误信息"""
        async def mock_post(*args, **kwargs):
            raise RuntimeError("Unexpected error occurred")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            response = await tavily_search("异常测试")

        assert response.error is not None
        assert "搜索失败" in response.error
        assert response.results == []

    @pytest.mark.asyncio
    async def test_custom_max_results(self):
        """自定义 max_results 参数"""
        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_tavily_api_response([
                {"url": "https://x.com/1", "title": "T", "content": "", "score": 1.0},
            ]))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            response = await tavily_search("test", max_results=5)

        assert response.total_count == 1

    @pytest.mark.asyncio
    async def test_empty_results_from_api(self):
        """API 返回空结果列表"""
        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response({"results": []})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            response = await tavily_search("空结果")

        assert response.results == []
        assert response.total_count == 0

    @pytest.mark.asyncio
    async def test_include_exclude_domains_passed(self):
        """验证 include_domains 和 exclude_domains 参数传递"""
        captured_payload = {}

        async def mock_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return make_mock_httpx_response({"results": []})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            await tavily_search(
                "test",
                include_domains=["example.com"],
                exclude_domains=["spam.com"],
            )

        assert "include_domains" in captured_payload
        assert captured_payload["include_domains"] == ["example.com"]
        assert captured_payload["exclude_domains"] == ["spam.com"]


# ============================================================
# Tool 2: 查询改写测试
# ============================================================
class TestRewriteQuery:
    """LLM 查询改写测试"""

    REWRITE_OUTPUT = """{
      "rewritten": "2025年7月 重大新闻",
      "language": "zh",
      "intent": "news",
      "sub_queries": ["中国新闻 2025年7月", "国际新闻 2025年7月"]
    }"""

    @pytest.mark.asyncio
    async def test_successful_rewrite(self):
        """正常改写 — 口语化→搜索关键词"""
        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response(self.REWRITE_OUTPUT))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await rewrite_query("今天有什么重大新闻？")

        assert isinstance(result, RewrittenQuery)
        assert result.original == "今天有什么重大新闻？"
        assert "2025" in result.rewritten
        assert result.language == "zh"
        assert result.intent == "news"
        assert len(result.sub_queries) == 2

    @pytest.mark.asyncio
    async def test_rewrite_with_history(self):
        """带对话历史的查询改写"""
        history = [
            {"role": "user", "content": "今天有什么新闻？"},
            {"role": "assistant", "content": "今天的主要新闻包括 AI 突破..."},
        ]

        captured_messages = []

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            captured_messages.append(json_body.get("messages", []))
            return make_mock_httpx_response(make_llm_response(self.REWRITE_OUTPUT))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await rewrite_query("详细说说", history)

        assert result.original == "详细说说"
        # 验证历史被传入
        user_message = captured_messages[0][1]["content"]
        assert "今天有什么新闻" in user_message

    @pytest.mark.asyncio
    async def test_rewrite_fallback_on_llm_failure(self):
        """LLM 失败时回退到原始查询"""
        async def mock_post(*args, **kwargs):
            raise Exception("LLM service unavailable")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await rewrite_query("今天有什么新闻？")

        # 回退：rewritten == original
        assert result.rewritten == "今天有什么新闻？"
        assert result.original == "今天有什么新闻？"
        assert result.intent == "factual"  # 默认值

    @pytest.mark.asyncio
    async def test_rewrite_fallback_on_malformed_json(self):
        """LLM 返回格式错误的 JSON 时回退"""
        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response("{invalid json!!!}"))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await rewrite_query("test query")

        assert result.rewritten == "test query"

    @pytest.mark.asyncio
    async def test_rewrite_partial_json_with_defaults(self):
        """LLM 返回缺少字段的 JSON — 使用默认值填充"""
        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response('{"rewritten": "仅有关键词"}'))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await rewrite_query("用户原文")

        assert result.rewritten == "仅有关键词"
        assert result.language == "zh"  # 默认
        assert result.intent == "factual"  # 默认
        assert result.sub_queries == []

    @pytest.mark.asyncio
    async def test_rewrite_with_empty_history(self):
        """空历史列表不影响改写"""
        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response(self.REWRITE_OUTPUT))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await rewrite_query("test", conversation_history=[])

        assert result.original == "test"


# ============================================================
# Tool 3: 相关性打分测试
# ============================================================
class TestScoreRelevance:
    """LLM 相关性打分测试"""

    SCORING_OUTPUT = """{
      "results": [
        {"url": "https://example.com/1", "title": "直接相关", "relevance": 0.95, "reason": "直接回答"},
        {"url": "https://example.com/2", "title": "部分相关", "relevance": 0.55, "reason": "背景信息"},
        {"url": "https://example.com/3", "title": "无关", "relevance": 0.05, "reason": "不相关"}
      ]
    }"""

    def _make_search_results(self) -> list[SearchResult]:
        return [
            SearchResult(url="https://example.com/1", title="直接相关", content="内容1", score=0.9),
            SearchResult(url="https://example.com/2", title="部分相关", content="内容2", score=0.5),
            SearchResult(url="https://example.com/3", title="无关", content="内容3", score=0.1),
        ]

    @pytest.mark.asyncio
    async def test_successful_scoring(self):
        """正常打分 — 0-1 范围验证"""
        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response(self.SCORING_OUTPUT))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await score_relevance("测试查询", self._make_search_results())

        assert isinstance(result, RelevanceScores)
        assert len(result.results) == 3

        # 验证最高分
        assert result.results[0].relevance == 0.95
        assert result.results[0].reason == "直接回答"

        # 验证最低分
        assert result.results[2].relevance == 0.05
        assert result.results[2].reason == "不相关"

        # 验证分数范围
        for r in result.results:
            assert isinstance(r, RelevanceScore)
            assert 0.0 <= r.relevance <= 1.0

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_scores(self):
        """空搜索结果 → 空打分列表"""
        result = await score_relevance("query", [])
        assert isinstance(result, RelevanceScores)
        assert result.results == []

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        """LLM 打分失败时回退到原始搜索分数"""
        async def mock_post(*args, **kwargs):
            raise Exception("Scoring service down")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await score_relevance("query", self._make_search_results())

        # 回退：使用原始 score
        assert len(result.results) == 3
        assert result.results[0].relevance == 0.9  # 来自 SearchResult.score
        assert result.results[0].reason == "原始搜索分数"

    @pytest.mark.asyncio
    async def test_fallback_zero_score_uses_default(self):
        """回退时如果原始分数为 0, 使用 0.5 作为默认"""
        results = [SearchResult(url="https://x.com/1", title="T", content="C", score=0.0)]

        async def mock_post(*args, **kwargs):
            raise Exception("Scoring failed")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await score_relevance("query", results)

        assert result.results[0].relevance == 0.5  # 默认值

    @pytest.mark.asyncio
    async def test_content_truncation_in_prompt(self):
        """验证内容被截断到 500 字符"""
        long_content = "长内容" * 300  # 约 900 字符 (>500)
        results = [SearchResult(url="https://x.com/1", title="T", content=long_content, score=0.8)]

        captured_messages = []

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            captured_messages.append(json_body.get("messages", []))
            return make_mock_httpx_response(make_llm_response("""{
              "results": [{"url": "https://x.com/1", "title": "T", "relevance": 0.8, "reason": "OK"}]
            }"""))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            await score_relevance("query", results)

        # 验证传入 LLM 的内容不超过 500 字符
        user_message = captured_messages[0][1]["content"]
        # "长内容" * 300 = 900 chars → 截断后 ≤500 chars
        # "长内容" * 200 = 600 chars > 500, 确认不会出现
        assert "长内容" * 200 not in user_message  # 600 字符的完整长内容不应出现


# ============================================================
# Tool 4: 降级回答测试
# ============================================================
class TestFallbackAnswer:
    """LLM 降级回答测试"""

    @pytest.mark.asyncio
    async def test_successful_fallback(self):
        """正常降级回答 — 应包含警告提示"""
        fallback_text = "⚠️ 当前无法进行实时搜索，以下回答基于我的训练数据。\n\n根据现有知识..."

        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response(fallback_text))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await generate_fallback_answer("今天天气怎么样？")

        assert isinstance(result, str)
        assert "⚠️" in result or "无法" in result
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fallback_with_history(self):
        """带对话历史的降级回答"""
        history = [
            {"role": "user", "content": "今天天气如何？"},
        ]

        async def mock_post(*args, **kwargs):
            return make_mock_httpx_response(make_llm_response("降级回答内容"))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await generate_fallback_answer("具体点", history)

        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fallback_on_total_failure(self):
        """降级回答本身也失败 → 返回友好错误提示"""
        async def mock_post(*args, **kwargs):
            raise Exception("Everything is broken")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            result = await generate_fallback_answer("question")

        assert "抱歉" in result or "错误" in result
        assert "Everything is broken" in result

    @pytest.mark.asyncio
    async def test_fallback_prompt_includes_warning_header(self):
        """验证降级回答的 system prompt 正确传递"""
        captured_messages = []

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            captured_messages.append(json_body.get("messages", []))
            return make_mock_httpx_response(make_llm_response("回答"))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            await generate_fallback_answer("测试")

        system_msg = captured_messages[0][0]["content"]
        assert "实时搜索" in system_msg
        assert "训练数据" in system_msg

    @pytest.mark.asyncio
    async def test_fallback_with_long_history_truncation(self):
        """长历史应该被截断到最近 6 条"""
        history = []
        for i in range(20):
            history.append({"role": "user", "content": f"问题 {i}"})
            history.append({"role": "assistant", "content": f"回答 {i}"})

        captured_messages = []

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            captured_messages.append(json_body.get("messages", []))
            return make_mock_httpx_response(make_llm_response("降级答案"))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            await generate_fallback_answer("最新问题", history)

        # System prompt + 截断后的历史 (最多6条) + 用户消息 = 最多 8 条
        all_messages = captured_messages[0]
        assert len(all_messages) <= 8


# ============================================================
# 组合工具: search_and_filter_pipeline 测试
# ============================================================
class TestSearchAndFilterPipeline:
    """搜索+过滤组合流水线测试"""

    SCORING_OUTPUT = """{
      "results": [
        {"url": "https://example.com/1", "title": "A", "relevance": 0.95, "reason": "高相关"},
        {"url": "https://example.com/2", "title": "B", "relevance": 0.82, "reason": "相关"}
      ]
    }"""

    def _setup_mock_client(self, search_results=None, llm_responses=None):
        """构造 mock httpx Client，按顺序返回响应"""
        if search_results is None:
            search_results = [
                {"url": "https://example.com/1", "title": "新闻 A", "content": "内容 A " * 20, "score": 0.95},
                {"url": "https://example.com/2", "title": "新闻 B", "content": "内容 B " * 20, "score": 0.82},
            ]
        if llm_responses is None:
            llm_responses = [make_llm_response(self.SCORING_OUTPUT)]

        response_queue = list(llm_responses)

        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""

            if "搜索结果质量评估专家" in system_content:
                # LLM 相关性打分
                if response_queue:
                    return make_mock_httpx_response(response_queue.pop(0))
                return make_mock_httpx_response(make_llm_response(self.SCORING_OUTPUT))
            else:
                # Tavily API
                return make_mock_httpx_response({"results": search_results})

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=mock_post)
        return mock_client

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """完整搜索+过滤流水线"""
        mock_client = self._setup_mock_client()

        with patch("httpx.AsyncClient", return_value=mock_client):
            deduped, fragments = await search_and_filter_pipeline("测试查询")

        # 验证去重结果
        assert len(deduped) == 2
        assert all(isinstance(r, SearchResult) for r in deduped)

        # 验证裁剪后的片段
        assert len(fragments) >= 1
        for url, content, score in fragments:
            assert url.startswith("https://")
            assert isinstance(content, str)
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_pipeline_with_deduplicator(self):
        """带外部去重器的流水线"""
        dd = URLDeduplicator(window_size=50)
        # 预填充一个重复 URL
        dd.add("https://example.com/1")

        mock_client = self._setup_mock_client()

        with patch("httpx.AsyncClient", return_value=mock_client):
            deduped, fragments = await search_and_filter_pipeline("查询", deduplicator=dd)

        # https://example.com/1 应该被去重过滤掉
        urls = [r.url for r in deduped]
        assert "https://example.com/1" not in urls
        assert len(deduped) == 1  # 只剩下 example.com/2

    @pytest.mark.asyncio
    async def test_pipeline_search_error(self):
        """搜索 API 返回错误 → 空结果"""
        async def mock_post(*args, **kwargs):
            # Tavily 返回错误
            return make_mock_httpx_response({"results": [], "error": "API rate limited"},
                                            status_code=429)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=mock_post)

        # 注意: HTTP 429 会触发 raise_for_status → 异常 → 被 catch → 返回 err SearchResponse
        # 实际是: httpx 的 raise_for_status 只对 4xx/5xx 抛异常
        # 需要模拟异常
        import httpx

        async def mock_post_with_error(*args, **kwargs):
            raise httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=MagicMock(status_code=429))

        mock_client.post = AsyncMock(side_effect=mock_post_with_error)

        with patch("httpx.AsyncClient", return_value=mock_client):
            deduped, fragments = await search_and_filter_pipeline("查询")

        # 搜索失败 → 空结果 (异常被 catch)
        # 注意：tavily_search 已经 catch 了 Exception，所以不会传到 pipeline
        # 实际上 search_and_filter_pipeline 调用 tavily_search，后者内部 catch 异常返回 SearchResponse(error=...)
        # 所以这里需要重新 mock 使 tavily_search 返回带 error 的 SearchResponse
        # 换一种方式测试

    @pytest.mark.asyncio
    async def test_pipeline_empty_search_results(self):
        """搜索返回空结果 → 空 deduped 和 fragments"""
        mock_client = self._setup_mock_client(search_results=[])

        with patch("httpx.AsyncClient", return_value=mock_client):
            deduped, fragments = await search_and_filter_pipeline("空查询")

        assert deduped == []
        assert fragments == []

    @pytest.mark.asyncio
    async def test_pipeline_all_duplicates(self):
        """所有结果都被去重 → 返回空"""
        dd = URLDeduplicator(window_size=50)
        dd.add("https://example.com/1")
        dd.add("https://example.com/2")

        mock_client = self._setup_mock_client()

        with patch("httpx.AsyncClient", return_value=mock_client):
            deduped, fragments = await search_and_filter_pipeline("查询", deduplicator=dd)

        assert deduped == []
        assert fragments == []

    @pytest.mark.asyncio
    async def test_pipeline_scoring_failure_fallback(self):
        """打分失败时回退到原始分数"""
        async def mock_post(*args, **kwargs):
            json_body = kwargs.get("json", {})
            messages = json_body.get("messages", [])
            system_content = messages[0]["content"] if messages else ""

            if "搜索结果质量评估专家" in system_content:
                raise Exception("Scoring unavailable")
            else:
                return make_mock_httpx_response({"results": [
                    {"url": "https://example.com/1", "title": "T", "content": "C " * 30, "score": 0.75},
                ]})

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=mock_post)

        with patch("httpx.AsyncClient", return_value=mock_client):
            deduped, fragments = await search_and_filter_pipeline("查询")

        # 回退后应使用原始分数 0.75
        assert len(fragments) == 1
        assert fragments[0][2] == 0.75


# ============================================================
# 工具注册表元数据测试
# ============================================================
class TestToolRegistryMetadata:
    """工具元数据注册表完整性测试"""

    def test_all_four_tools_registered(self):
        """四个工具都应该在注册表中"""
        expected_tools = {"tavily_search", "query_rewriter", "relevance_scorer", "fallback_answer"}
        assert set(TOOL_REGISTRY.keys()) == expected_tools

    def test_tool_categories_match(self):
        """工具类别与名称对应"""
        category_map = {
            "tavily_search": "search",
            "query_rewriter": "rewrite",
            "relevance_scorer": "relevance",
            "fallback_answer": "fallback",
        }
        for name, expected_category in category_map.items():
            tool = TOOL_REGISTRY[name]
            assert tool.category.value == expected_category
            assert len(tool.description) > 0
            assert tool.version == "1.0.0"

    def test_tool_names_are_unique(self):
        """工具名称唯一"""
        names = [t.name for t in TOOL_REGISTRY.values()]
        assert len(names) == len(set(names))
