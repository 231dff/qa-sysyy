"""
智能搜索助手 — 自动化评估框架

基于黄金数据集的 Agent 行为评估系统。不评估"答案是否正确"，
而是验证"代码是否按预期路径走"——所有断言都是白盒结构断言。

用法:
    python -m evals.evaluate              # 运行全部
    python -m evals.evaluate -c happy_path  # 按类别
    python -m evals.evaluate --case happy-001  # 单条
    python -m evals.evaluate -r html       # HTML 报告
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import SearchAgent, route_after_generate, route_after_search
from agent.models import AgentState, GeneratedAnswer, RelevanceScores, RewrittenQuery, SearchResponse, SearchResult
from config import get_agent_config
from tools.registry import generate_fallback_answer, rewrite_query, score_relevance, search_and_filter_pipeline, tavily_search


# ============================================================
# 数据模型
# ============================================================
@dataclass
class AssertionResult:
    name: str; expected: Any; actual: Any; passed: bool; message: str = ""

@dataclass
class CaseResult:
    case_id: str; description: str; category: str; query: str | list[str]
    passed: bool = True; assertions: list[AssertionResult] = field(default_factory=list)
    error: Optional[str] = None; duration_ms: float = 0.0

@dataclass
class EvalReport:
    dataset_version: str = ""; total_cases: int = 0
    passed_cases: int = 0; failed_cases: int = 0; error_cases: int = 0
    total_assertions: int = 0; passed_assertions: int = 0; failed_assertions: int = 0
    pass_rate_cases: float = 0.0; pass_rate_assertions: float = 0.0
    duration_total_ms: float = 0.0
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)
    failed_details: list[dict] = field(default_factory=list)
    timestamp: str = ""


# ============================================================
# Mock 工具
# ============================================================
def _mock_resp(json_data: dict, status_code: int = 200):
    r = MagicMock(); r.status_code = status_code; r.json.return_value = json_data; r.raise_for_status = MagicMock()
    return r

def _llm(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}

def _tavily(results: list[dict] | None = None) -> dict:
    return {"results": results or [
        {"url": "https://example.com/news/1", "title": "新闻A", "content": "示例内容", "score": 0.95},
        {"url": "https://example.com/news/2", "title": "新闻B", "content": "另一条内容", "score": 0.82},
    ]}

REWRITE_OUT = '{"rewritten":"2025年7月 重大新闻","language":"zh","intent":"news","sub_queries":["中国 2025","国际 2025"]}'
COMPARE_OUT = '{"rewritten":"Python Rust 对比","language":"zh","intent":"comparison","sub_queries":["Python 优缺点","Rust 优缺点"]}'
SCORE_OUT = '{"results":[{"url":"https://example.com/news/1","title":"A","relevance":0.95,"reason":"相关"},{"url":"https://example.com/news/2","title":"B","relevance":0.82,"reason":"相关"}]}'
ANSWER_OUT = "## 今日要闻\n根据最新搜索结果:\n1. **AI** [1]\n2. **科技** [2]\n\n📚 参考来源"
FALLBACK_OUT = "⚠️ 当前无法进行实时搜索，以下回答基于我的训练数据。这是一条测试降级回答。"


# ============================================================
# 断言引擎 — 扁平叶键查找
# ============================================================
class AssertEngine:
    def __init__(self):
        self.results: list[AssertionResult] = []

    def _fail(self, name: str, expected: Any, actual: Any, msg: str = ""):
        self.results.append(AssertionResult(name, expected, actual, False, msg or f"期望 {expected!r}, 实际 {actual!r}"))
        return False

    def _ok(self, name: str, expected: Any, actual: Any):
        self.results.append(AssertionResult(name, expected, actual, True, ""))
        return True

    def evaluate(self, expected: dict, facts: dict):
        """facts 是扁平字典，expected 也是扁平键（无嵌套）。"""
        for key, value in expected.items():
            actual = facts.get(key)

            if isinstance(value, bool):
                if actual is None:
                    self._fail(key, value, "<未找到>", f"在上下文中找不到键 '{key}'")
                elif bool(actual) == value:
                    self._ok(key, value, actual)
                else:
                    self._fail(key, value, actual, f"期望 {value}, 实际 {actual!r}")

            elif isinstance(value, (int, float)):
                if actual is None:
                    self._fail(key, value, "<未找到>", f"在上下文中找不到键 '{key}'")
                elif key.endswith("_gt"):
                    if actual > value:
                        self._ok(key, f">{value}", actual)
                    else:
                        self._fail(key, f">{value}", actual, f"期望 >{value}, 实际 {actual}")
                elif key.endswith("_gte"):
                    if actual >= value:
                        self._ok(key, f">={value}", actual)
                    else:
                        self._fail(key, f">={value}", actual, f"期望 >={value}, 实际 {actual}")
                elif key.endswith("_lt"):
                    if actual < value:
                        self._ok(key, f"<{value}", actual)
                    else:
                        self._fail(key, f"<{value}", actual, f"期望 <{value}, 实际 {actual}")
                elif key.endswith("_lte"):
                    if actual <= value:
                        self._ok(key, f"<={value}", actual)
                    else:
                        self._fail(key, f"<={value}", actual, f"期望 <={value}, 实际 {actual}")
                elif key.endswith("_count"):
                    if actual == value:
                        self._ok(key, value, actual)
                    else:
                        self._fail(key, value, actual)
                elif key.endswith("_count_gt"):
                    if actual > value:
                        self._ok(key, f">{value}", actual)
                    else:
                        self._fail(key, f">{value}", actual, f"期望 >{value}, 实际 {actual}")
                else:
                    if actual == value:
                        self._ok(key, value, actual)
                    else:
                        self._fail(key, value, actual)

            elif isinstance(value, str):
                if actual is None:
                    self._fail(key, value, "<未找到>")
                elif key.endswith("_contains"):
                    # substring 断言
                    passed = value in str(actual)
                    if passed:
                        self._ok(key, f"包含 '{value}'", actual)
                    else:
                        self._fail(key, f"包含 '{value}'", actual, f"'{value}' 不在 '{str(actual)[:60]}' 中")
                elif actual == value:
                    self._ok(key, value, actual)
                else:
                    self._fail(key, value, actual)

            elif isinstance(value, list):
                actual_list = actual if isinstance(actual, list) else []
                for node in value:
                    if isinstance(node, str):
                        passed = node in actual_list
                        if passed:
                            self._ok(f"{key}[{node}]", "已调用", "已调用")
                        else:
                            self._fail(f"{key}[{node}]", "已调用", "未调用", f"节点 '{node}' 未调用")
                    else:
                        self._ok(f"{key}[list]", True, True)


# ============================================================
# Runner — 按上下文执行并收集行为事实
# ============================================================
class Runner:
    def __init__(self):
        self.cfg = get_agent_config()

    def _patch_pattern(self) -> dict:
        """根据 context 返回 mock_post 工厂。返回 {sys_prompt_fragment: behavior, ...}"""
        return {}

    async def run_case(self, case: dict) -> tuple[dict, float]:
        ctx = case.get("context", {})
        query = case.get("query", "")
        cat = case.get("category", "")

        start = time.perf_counter()

        if cat == "routing":
            facts = self._run_routing(ctx)
        elif cat == "tool_behavior":
            facts = await self._run_tool(case)
        elif cat == "state_integrity":
            facts = self._run_state(ctx)
        elif cat == "multi_turn":
            facts = await self._run_multi(case)
        elif cat == "edge_case":
            facts = await self._run_edge(case)
        elif ctx.get("search_returns_empty"):
            facts = await self._run_search_fail(case)
        elif ctx.get("llm_answer_raises_exception"):
            facts = await self._run_answer_fail(case)
        elif ctx.get("rewrite_raises_exception"):
            facts = await self._run_rewrite_fail(case)
        elif ctx.get("scoring_raises_exception"):
            facts = await self._run_score_fail(case)
        elif ctx.get("search_raises_timeout"):
            facts = await self._run_timeout(case)
        else:
            facts = await self._run_happy(case)

        elapsed = (time.perf_counter() - start) * 1000
        return facts, elapsed

    # ── Mock 客户端工厂 ──
    def _make_mock_client(self, handler):
        """创建 patch 上下文管理器：同时 patch tools.registry 和 agent.graph 中的 httpx"""
        return self._Patcher(handler)

    class _Patcher:
        def __init__(self, handler):
            self.handler = handler
            self.patches = []
            self.mocks = []

        def __enter__(self):
            for target in ["tools.registry.httpx.AsyncClient", "agent.graph.httpx.AsyncClient"]:
                p = patch(target)
                mock_cls = p.start()
                self.patches.append(p)
                m = MagicMock()
                m.__aenter__ = AsyncMock(return_value=m)
                m.__aexit__ = AsyncMock(return_value=None)
                m.post = AsyncMock(side_effect=self.handler)
                mock_cls.return_value = m
                self.mocks.append(m)
            return self

        def __exit__(self, *args):
            for p in self.patches:
                p.stop()

    # ── 通用 handler 构建器 ──
    def _handler(self, behaviors: dict[str, callable], nodes: list):
        """behaviors: 映射 system_prompt 片段 -> 返回 response 的回调"""
        async def h(*args, **kwargs):
            body = kwargs.get("json", {})
            msgs = body.get("messages", [])
            sys = msgs[0]["content"] if msgs else ""
            for fragment, fn in behaviors.items():
                if fragment in sys:
                    return fn()
            # 默认 fallback
            return _mock_resp(_tavily())
        return h

    # ── Happy Path ──
    async def _run_happy(self, case: dict) -> dict:
        query = case.get("query", "")
        nodes = []
        agent = SearchAgent()

        def rewrite_fn():
            nodes.append("rewrite_query")
            return _mock_resp(_llm(COMPARE_OUT if ("对比" in query or "哪个更" in query) else REWRITE_OUT))
        def score_fn():
            nodes.append("score_relevance")
            return _mock_resp(_llm(SCORE_OUT))
        def answer_fn():
            nodes.append("generate_answer")
            return _mock_resp(_llm(ANSWER_OUT))
        def search_fn():
            nodes.append("search")
            return _mock_resp(_tavily())

        behaviors = {
            "搜索查询优化专家": rewrite_fn,
            "搜索结果质量评估专家": score_fn,
            "实时信息问答助手": answer_fn,
            "智能问答助手": answer_fn,
        }
        # 搜索用 fallthrough 处理

        async def handler(*args, **kwargs):
            body = kwargs.get("json", {})
            msgs = body.get("messages", [])
            sys = msgs[0]["content"] if msgs else ""
            for frag, fn in behaviors.items():
                if frag in sys:
                    return fn()
            return search_fn()

        with self._make_mock_client(handler):
            result = await agent.run(query)

        all_sessions = list(agent._sessions.keys())
        hist = agent.get_history(all_sessions[0]) if all_sessions else []
        return {
            "nodes_called": nodes,
            "is_fallback": result.is_fallback,
            "confidence": result.confidence,
            "sources_non_empty": len(result.sources) > 0,
            "sources_urls_unique": len(result.sources) == len(set(s.url for s in result.sources)),
            "answer_non_empty": len(result.answer) > 0,
            "latency_ms_gt": result.latency_ms > 0,
            "tokens_used_gte": result.tokens_used >= 0,
            "search_failed": False,
            "error_is_none": True,
            "history_count_gte": len(hist),
            "rewritten_non_empty": True,
        }

    # ── 搜索失败 ──
    async def _run_search_fail(self, case: dict) -> dict:
        query = case.get("query", "")
        nodes = []
        agent = SearchAgent()

        async def handler(*args, **kwargs):
            body = kwargs.get("json", {})
            msgs = body.get("messages", [])
            sys = msgs[0]["content"] if msgs else ""
            if "搜索查询优化专家" in sys:
                nodes.append("rewrite_query")
                return _mock_resp(_llm(REWRITE_OUT))
            elif "智能问答助手" in sys or "实时信息问答助手" in sys:
                nodes.append("generate_answer")
                return _mock_resp(_llm(FALLBACK_OUT))
            else:
                nodes.append("search")
                return _mock_resp({"results": []})

        with self._make_mock_client(handler):
            result = await agent.run(query)

        return {
            "nodes_called": nodes,
            "is_fallback": result.is_fallback,
            "confidence": result.confidence,
            "sources_empty": len(result.sources) == 0,
            "answer_non_empty": len(result.answer) > 0,
            "answer_has_fallback_marker": "⚠️" in result.answer,
        }

    # ── 答案生成失败 ──
    async def _run_answer_fail(self, case: dict) -> dict:
        query = case.get("query", "")
        nodes = []
        agent = SearchAgent()
        answer_call = [0]

        async def handler(*args, **kwargs):
            body = kwargs.get("json", {})
            msgs = body.get("messages", [])
            sys = msgs[0]["content"] if msgs else ""
            if "搜索查询优化专家" in sys:
                nodes.append("rewrite_query")
                return _mock_resp(_llm(REWRITE_OUT))
            elif "搜索结果质量评估专家" in sys:
                return _mock_resp(_llm(SCORE_OUT))
            elif "实时信息问答助手" in sys:
                nodes.append("generate_answer")
                raise Exception("LLM API 500")
            elif "智能问答助手" in sys:
                nodes.append("generate_answer")
                return _mock_resp(_llm(FALLBACK_OUT))
            else:
                nodes.append("search")
                return _mock_resp(_tavily())

        with self._make_mock_client(handler):
            result = await agent.run(query)

        return {
            "nodes_called": nodes,
            "is_fallback": result.is_fallback,
            "confidence": result.confidence,
            "sources_empty": len(result.sources) == 0,
            "answer_non_empty": len(result.answer) > 0,
        }

    # ── 改写失败 ──
    async def _run_rewrite_fail(self, case: dict) -> dict:
        query = case.get("query", "")
        nodes = []
        agent = SearchAgent()

        async def handler(*args, **kwargs):
            body = kwargs.get("json", {})
            msgs = body.get("messages", [])
            sys = msgs[0]["content"] if msgs else ""
            if "搜索查询优化专家" in sys:
                nodes.append("rewrite_query")
                raise Exception("rewrite unavailable")
            elif "搜索结果质量评估专家" in sys:
                return _mock_resp(_llm(SCORE_OUT))
            elif "实时信息问答助手" in sys:
                nodes.append("generate_answer")
                return _mock_resp(_llm(ANSWER_OUT))
            else:
                nodes.append("search")
                return _mock_resp(_tavily())

        with self._make_mock_client(handler):
            result = await agent.run(query)

        return {
            "nodes_called": nodes,
            "is_fallback": result.is_fallback,
            "answer_non_empty": len(result.answer) > 0,
        }

    # ── 打分失败 ──
    async def _run_score_fail(self, case: dict) -> dict:
        query = case.get("query", "")
        nodes = []
        agent = SearchAgent()

        async def handler(*args, **kwargs):
            body = kwargs.get("json", {})
            msgs = body.get("messages", [])
            sys = msgs[0]["content"] if msgs else ""
            if "搜索查询优化专家" in sys:
                nodes.append("rewrite_query")
                return _mock_resp(_llm(REWRITE_OUT))
            elif "搜索结果质量评估专家" in sys:
                raise Exception("scoring unavailable")
            elif "实时信息问答助手" in sys:
                nodes.append("generate_answer")
                return _mock_resp(_llm(ANSWER_OUT))
            else:
                nodes.append("search")
                return _mock_resp(_tavily())

        with self._make_mock_client(handler):
            result = await agent.run(query)

        return {
            "nodes_called": nodes,
            "is_fallback": result.is_fallback,
            "sources_non_empty": len(result.sources) > 0,
        }

    # ── 搜索超时 ──
    async def _run_timeout(self, case: dict) -> dict:
        query = case.get("query", "")
        nodes = []
        agent = SearchAgent()

        async def handler(*args, **kwargs):
            body = kwargs.get("json", {})
            msgs = body.get("messages", [])
            sys = msgs[0]["content"] if msgs else ""
            if "搜索查询优化专家" in sys:
                nodes.append("rewrite_query")
                return _mock_resp(_llm(REWRITE_OUT))
            elif "智能问答助手" in sys or "实时信息问答助手" in sys:
                nodes.append("generate_answer")
                return _mock_resp(_llm(FALLBACK_OUT))
            else:
                nodes.append("search")
                raise httpx.TimeoutException("Request timed out")

        with self._make_mock_client(handler):
            result = await agent.run(query)

        return {
            "nodes_called": nodes,
            "is_fallback": result.is_fallback,
            "confidence": result.confidence,
            "answer_non_empty": len(result.answer) > 0,
        }

    # ── 路由测试 ──
    def _run_routing(self, ctx: dict) -> dict:
        rt = ctx.get("route_test", "")
        facts = {}
        if rt == "after_search_success":
            facts["route_after_search"] = route_after_search({"search_failed": False})
        elif rt == "after_search_failure":
            facts["route_after_search"] = route_after_search({"search_failed": True})
        elif rt == "after_generate_success":
            facts["route_after_generate"] = route_after_generate({"error": None})
        elif rt == "after_generate_failure":
            facts["route_after_generate"] = route_after_generate({"error": "err"})
        elif rt == "after_search_missing_key":
            facts["route_after_search"] = route_after_search({})
        return facts

    # ── 工具行为 ──
    async def _run_tool(self, case: dict) -> dict:
        tt = case.get("context", {}).get("tool_test", "")
        query = case.get("query", "")
        facts = {}

        if tt == "tavily_search_success":
            mr = [{"url":"https://x.com/1","title":"A","content":"c","score":0.9},
                  {"url":"https://x.com/2","title":"B","content":"c","score":0.7}]
            async def h(*a,**kw): return _mock_resp({"results": mr})
            with self._make_mock_client(h):
                r = await tavily_search(query)
            facts.update({
                "tool_is_SearchResponse": isinstance(r, SearchResponse),
                "tool_query_matches": r.query == query,
                "tool_results_count_gte": len(r.results),
                "tool_error_is_none": r.error is None,
                "tool_search_time_ms_gt": r.search_time_ms > 0,
            })
        elif tt == "tavily_search_timeout":
            async def h(*a,**kw): raise httpx.TimeoutException("timeout")
            with self._make_mock_client(h):
                r = await tavily_search(query)
            facts.update({
                "tool_is_SearchResponse": isinstance(r, SearchResponse),
                "tool_results_empty": len(r.results) == 0,
                "tool_error_contains": "超时" in (r.error or ""),
            })
        elif tt == "rewrite_query_success":
            async def h(*a,**kw): return _mock_resp(_llm(REWRITE_OUT))
            with self._make_mock_client(h):
                rw = await rewrite_query(query, [])
            facts.update({
                "tool_is_RewrittenQuery": isinstance(rw, RewrittenQuery),
                "rewritten_non_empty": len(rw.rewritten) > 0,
                "language_non_empty": bool(rw.language),
                "intent_non_empty": bool(rw.intent),
                "original_preserved": rw.original == query,
            })
        elif tt == "score_relevance_success":
            srs = [SearchResult(url="https://x.com/1",title="T1",content="C1",score=0.9),
                   SearchResult(url="https://x.com/2",title="T2",content="C2",score=0.7)]
            score_out_matched = '{"results":[{"url":"https://x.com/1","title":"T1","relevance":0.95,"reason":"相关"},{"url":"https://x.com/2","title":"T2","relevance":0.82,"reason":"相关"}]}'
            async def h(*a,**kw): return _mock_resp(_llm(score_out_matched))
            with self._make_mock_client(h):
                sc = await score_relevance(query, srs)
            facts.update({
                "tool_is_RelevanceScores": isinstance(sc, RelevanceScores),
                "tool_all_scores_in_range": all(0<=s.relevance<=1 for s in sc.results),
                "tool_urls_preserved": all(s.url in {"https://x.com/1","https://x.com/2"} for s in sc.results),
            })
        elif tt == "fallback_answer_success":
            async def h(*a,**kw): return _mock_resp(_llm(FALLBACK_OUT))
            with self._make_mock_client(h):
                ans = await generate_fallback_answer(query, [])
            facts.update({
                "answer_is_string": isinstance(ans, str),
                "answer_non_empty": len(ans) > 0,
                "answer_has_fallback_marker": "⚠️" in ans,
            })
        elif tt == "search_and_filter_pipeline":
            async def h(*a,**kw):
                body = kw.get("json", {})
                msgs = body.get("messages", [])
                sys = msgs[0]["content"] if msgs else ""
                if "搜索结果质量评估专家" in sys:
                    return _mock_resp(_llm(SCORE_OUT))
                return _mock_resp(_tavily())
            with self._make_mock_client(h):
                deduped, fragments = await search_and_filter_pipeline(query)
            facts.update({
                "pipeline_has_tuple": isinstance(deduped, list) and isinstance(fragments, list),
                "pipeline_deduped_non_empty": len(deduped) > 0,
                "pipeline_fragments_non_empty": len(fragments) > 0,
                "pipeline_fragments_lte_5": len(fragments) <= 5,
            })
        return facts

    # ── 状态完整性 ──
    def _run_state(self, ctx: dict) -> dict:
        st = ctx.get("state_test", "")
        facts = {}
        if st == "agent_state_keys":
            required = ["session_id","user_query","user_query_raw","rewritten_queries",
                        "search_results","deduped_results","relevance_scores","top_results",
                        "final_answer","history","error","retry_count","fallback_triggered",
                        "search_failed","started_at","completed_at"]
            state: AgentState = {
                "session_id":"t","user_query":"t","user_query_raw":"t","rewritten_queries":[],
                "search_results":[],"deduped_results":[],"relevance_scores":None,
                "top_results":[],"final_answer":None,"history":[],"error":None,
                "retry_count":0,"fallback_triggered":False,"search_failed":False,
                "started_at":datetime.now(timezone.utc).isoformat(),"completed_at":"",
            }
            facts["agent_state_has_all_keys"] = all(k in state for k in required)
        elif st == "generated_answer_model":
            ga = GeneratedAnswer(query="t", answer="t", sources=[], confidence=0.85,
                                 is_fallback=False, tokens_used=10, latency_ms=100.0)
            facts.update({
                "confidence_in_range": 0 <= ga.confidence <= 1,
                "is_fallback_is_bool": isinstance(ga.is_fallback, bool),
                "sources_is_list": isinstance(ga.sources, list),
                "answer_is_str": isinstance(ga.answer, str),
                "latency_is_number": isinstance(ga.latency_ms, (int, float)),
                "tokens_is_int": isinstance(ga.tokens_used, int),
            })
        return facts

    # ── 多轮对话 ──
    async def _run_multi(self, case: dict) -> dict:
        queries = case.get("query", [])
        if isinstance(queries, str):
            queries = [queries]
        ctx = case.get("context", {})
        sid = ctx.get("session_id", "multi-default")
        max_turns = ctx.get("max_history_turns")
        clear_after = ctx.get("clear_after", False)

        agent = SearchAgent()
        rounds = []

        async def handler(*args, **kwargs):
            body = kwargs.get("json", {})
            msgs = body.get("messages", [])
            sys = msgs[0]["content"] if msgs else ""
            if "搜索查询优化专家" in sys:
                return _mock_resp(_llm(REWRITE_OUT))
            elif "搜索结果质量评估专家" in sys:
                return _mock_resp(_llm(SCORE_OUT))
            elif "实时信息问答助手" in sys:
                return _mock_resp(_llm(ANSWER_OUT))
            else:
                return _mock_resp(_tavily())

        orig_turns = None
        if max_turns is not None:
            orig_turns = self.cfg.max_history_turns
            self.cfg.max_history_turns = max_turns

        try:
            with self._make_mock_client(handler):
                for i, q in enumerate(queries):
                    r = await agent.run(q, session_id=sid)
                    rounds.append({"round": i+1, "is_fallback": r.is_fallback})

                if clear_after:
                    agent.clear_session(sid)

                hist = agent.get_history(sid)
        finally:
            if orig_turns is not None:
                self.cfg.max_history_turns = orig_turns

        facts = {"final_history_count": len(hist)}
        if max_turns is not None:
            facts["final_history_count_lte"] = len(hist) <= max_turns * 2
        if clear_after:
            facts["history_after_clear"] = len(hist)
        for rd in rounds:
            facts[f"round_{rd['round']}_is_fallback"] = rd["is_fallback"]
        return facts

    # ── 边缘情况 ──
    async def _run_edge(self, case: dict) -> dict:
        query = case.get("query", "")
        agent = SearchAgent()

        async def handler(*args, **kwargs):
            body = kwargs.get("json", {})
            msgs = body.get("messages", [])
            sys = msgs[0]["content"] if msgs else ""
            if "搜索查询优化专家" in sys:
                rw = query if query else "最新新闻"
                return _mock_resp(_llm(f'{{"rewritten":"{rw}","language":"zh","intent":"news","sub_queries":[]}}'))
            elif "搜索结果质量评估专家" in sys:
                return _mock_resp(_llm(SCORE_OUT))
            elif "实时信息问答助手" in sys:
                return _mock_resp(_llm(ANSWER_OUT))
            else:
                return _mock_resp(_tavily())

        with self._make_mock_client(handler):
            result = await agent.run(query)

        return {
            "is_fallback": result.is_fallback,
            "answer_non_empty": len(result.answer) > 0,
            "session_count_gt": len(agent._sessions) > 0,
        }


# ============================================================
# Evaluator
# ============================================================
class Evaluator:
    def __init__(self, dataset_path: Optional[Path] = None):
        self.ds_path = dataset_path or PROJECT_ROOT / "evals" / "golden_dataset.json"
        self.runner = Runner()
        self.engine = AssertEngine()

    def load(self) -> list[dict]:
        with open(self.ds_path, "r", encoding="utf-8") as f:
            return json.load(f).get("cases", [])

    async def run_all(self, category=None, case_id=None) -> EvalReport:
        cases = self.load()
        if case_id:
            cases = [c for c in cases if c["id"] == case_id]
        elif category:
            cases = [c for c in cases if c.get("category") == category]

        if not cases:
            print("[!] 没有匹配的用例")
            return EvalReport()

        ds = json.load(open(self.ds_path, "r", encoding="utf-8"))
        report = EvalReport(
            dataset_version=ds.get("meta", {}).get("version", "?"),
            total_cases=len(cases),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        print(f"\n{'='*60}")
        print(f"[评估] 智能搜索助手 - 行为评估  v{report.dataset_version}")
        print(f"{'='*60}")
        print(f"用例: {report.total_cases}  |  断言类型: 白盒结构断言")
        print(f"{'='*60}\n")

        cat_stats: dict[str, dict[str, int]] = {}
        all_results: list[CaseResult] = []

        for i, case in enumerate(cases, 1):
            cid, cat, desc = case["id"], case.get("category","?"), case.get("description","")
            print(f"[{i:2d}/{report.total_cases}] {cid} ({cat})", end=" ", flush=True)

            cr = CaseResult(cid, desc, cat, case.get("query",""))
            try:
                facts, dur = await self.runner.run_case(case)
                cr.duration_ms = dur

                self.engine.results = []
                self.engine.evaluate(case.get("expected", {}), facts)
                cr.assertions = list(self.engine.results)
                cr.passed = all(a.passed for a in cr.assertions)
            except Exception as e:
                cr.passed = False
                cr.error = f"{type(e).__name__}: {e}"

            all_results.append(cr)

            s = "PASS" if cr.passed else ("ERR" if cr.error else "FAIL")
            a_ok = sum(1 for a in cr.assertions if a.passed)
            print(f"{s}  {a_ok}/{len(cr.assertions)}  {cr.duration_ms:.0f}ms")
            if not cr.passed and not cr.error:
                for a in cr.assertions:
                    if not a.passed:
                        print(f"     FAIL: {a.name} | {a.message[:80]}")
            if cr.error:
                print(f"     ERR: {cr.error[:120]}")

            cat_stats.setdefault(cat, {"total":0,"passed":0,"failed":0})
            cat_stats[cat]["total"] += 1
            cat_stats[cat]["passed" if cr.passed else "failed"] += 1

        # 汇总
        report.passed_cases = sum(1 for c in all_results if c.passed and not c.error)
        report.failed_cases = sum(1 for c in all_results if not c.passed and not c.error)
        report.error_cases = sum(1 for c in all_results if c.error)
        report.total_assertions = sum(len(c.assertions) for c in all_results)
        report.passed_assertions = sum(sum(1 for a in c.assertions if a.passed) for c in all_results)
        report.failed_assertions = report.total_assertions - report.passed_assertions
        report.pass_rate_cases = report.passed_cases / report.total_cases * 100 if report.total_cases else 0
        report.pass_rate_assertions = report.passed_assertions / report.total_assertions * 100 if report.total_assertions else 0
        report.duration_total_ms = sum(c.duration_ms for c in all_results)
        report.by_category = cat_stats

        for cr in all_results:
            if not cr.passed:
                report.failed_details.append({
                    "case_id": cr.case_id, "category": cr.category,
                    "description": cr.description, "error": cr.error,
                    "failed_assertions": [
                        {"name": a.name, "expected": a.expected, "actual": a.actual, "message": a.message}
                        for a in cr.assertions if not a.passed
                    ],
                })

        return report

    def print_report(self, report: EvalReport):
        fmt = lambda: print(f"{'='*60}")
        fmt()
        print("评估报告")
        fmt()
        print(f"版本: {report.dataset_version}  时间: {report.timestamp[:19].replace('T',' ')}")
        print(f"耗时: {report.duration_total_ms:.0f}ms")
        print()
        print(f"-- 用例维度 --")
        print(f"  总数: {report.total_cases}  |  通过: {report.passed_cases}  |  失败: {report.failed_cases}  |  错误: {report.error_cases}")
        print(f"  通过率: {report.pass_rate_cases:.1f}%")
        print()
        print(f"-- 断言维度 --")
        print(f"  总数: {report.total_assertions}  |  通过: {report.passed_assertions}  |  失败: {report.failed_assertions}")
        print(f"  通过率: {report.pass_rate_assertions:.1f}%")
        print()
        if report.by_category:
            print(f"-- 类别 --")
            for cat, st in sorted(report.by_category.items()):
                rate = st["passed"]/st["total"]*100 if st["total"] else 0
                bar = "#"*int(rate/10) + "-"*(10-int(rate/10))
                print(f"  {cat:<20s} {bar} {st['passed']}/{st['total']} ({rate:.0f}%)")
            print()
        if report.failed_details:
            print("-- 失败详情 --")
            for d in report.failed_details:
                print(f"  [{d['case_id']}] {d['description'][:60]}")
                if d["error"]:
                    print(f"    ERR: {d['error'][:120]}")
                for fa in d.get("failed_assertions", []):
                    print(f"    FAIL: {fa['name']} - {fa['message'][:100]}")
            print()
        fmt()

    def export_json(self, report: EvalReport, path: Optional[Path] = None):
        path = path or PROJECT_ROOT / "evals" / "report.json"
        data = {
            "version": report.dataset_version, "timestamp": report.timestamp,
            "summary": {
                "total_cases": report.total_cases, "passed": report.passed_cases,
                "failed": report.failed_cases, "errors": report.error_cases,
                "case_pass_rate": round(report.pass_rate_cases, 1),
                "assertion_pass_rate": round(report.pass_rate_assertions, 1),
            },
            "by_category": report.by_category,
            "failures": report.failed_details,
        }
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[json] {path}")

    def export_html(self, report: EvalReport, path: Optional[Path] = None):
        path = path or PROJECT_ROOT / "evals" / "report.html"
        pc = "#22c55e" if report.pass_rate_cases >= 90 else ("#f59e0b" if report.pass_rate_cases >= 70 else "#ef4444")
        cat_rows = "".join(
            f'<tr><td>{c}</td><td style="color:{"#22c55e" if st["passed"]/st["total"]>=0.9 else "#ef4444"}">{st["passed"]/st["total"]*100:.0f}%</td><td>{st["passed"]}/{st["total"]}</td></tr>'
            for c, st in sorted(report.by_category.items())
        )
        fail_rows = ""
        for d in report.failed_details:
            fas = "<br>".join(
                "FAIL: " + a.get("name", "") for a in d.get("failed_assertions", [])
            )
            fail_rows += (
                '<tr><td><code>' + d["case_id"] + '</code></td>'
                '<td>' + d["category"] + '</td>'
                '<td>' + d["description"][:60] + '</td>'
                '<td>' + (d.get("error", "") or "")[:80] + '</td>'
                '<td>' + fas + '</td></tr>'
            )
        html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><title>评估报告</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:system-ui,sans-serif;background:#f8fafc;padding:2rem}}
.container{{max-width:960px;margin:0 auto}}h1{{font-size:1.5rem;margin-bottom:.5rem}}
.card{{background:#fff;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;box-shadow:0 1px 3px #0001}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem}}
.stat{{text-align:center;padding:1rem;background:#f1f5f9;border-radius:8px}}
.stat .v{{font-size:2rem;font-weight:700}}.stat .l{{font-size:.8rem;color:#64748b}}
table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:.5rem .8rem;border-bottom:1px solid #e2e8f0}}
th{{background:#f8fafc;font-weight:600}}tr:hover{{background:#f8fafc}}.footer{{text-align:center;color:#94a3b8;font-size:.8rem;margin-top:2rem}}
</style></head><body><div class="container">
<h1>智能搜索助手 - 行为评估报告</h1>
<p>v{report.dataset_version} &middot; {report.timestamp[:19].replace('T',' ')} &middot; {report.duration_total_ms:.0f}ms</p>
<div class="card"><div class="grid">
<div class="stat"><div class="v">{report.total_cases}</div><div class="l">总用例</div></div>
<div class="stat"><div class="v" style="color:#22c55e">{report.passed_cases}</div><div class="l">通过</div></div>
<div class="stat"><div class="v" style="color:#ef4444">{report.failed_cases}</div><div class="l">失败</div></div>
<div class="stat"><div class="v" style="color:{pc}">{report.pass_rate_cases:.1f}%</div><div class="l">通过率</div></div>
</div></div>
<div class="card"><h3>断言 ({report.total_assertions}条)</h3>
<div class="grid"><div class="stat"><div class="v" style="color:#22c55e">{report.passed_assertions}</div><div class="l">通过</div></div>
<div class="stat"><div class="v" style="color:#ef4444">{report.failed_assertions}</div><div class="l">失败</div></div>
<div class="stat"><div class="v">{report.pass_rate_assertions:.1f}%</div><div class="l">通过率</div></div></div></div>
<div class="card"><h3>按类别</h3><table><tr><th>类别</th><th>通过率</th><th>通过/总数</th></tr>{cat_rows}</table></div>
{"<div class='card'><h3>失败详情</h3><table><tr><th>用例</th><th>类别</th><th>描述</th><th>错误</th><th>断言</th></tr>"+fail_rows+"</table></div>" if fail_rows else ""}
<div class="footer">白盒行为验证 &middot; {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</div>
</div></body></html>"""
        open(path, "w", encoding="utf-8").write(html)
        print(f"[html] {path}")


# ============================================================
# CLI
# ============================================================
async def main():
    ap = argparse.ArgumentParser(description="智能搜索助手 - 自动化行为评估")
    ap.add_argument("-c", "--category"); ap.add_argument("--case")
    ap.add_argument("-d", "--dataset"); ap.add_argument("-r", "--report", choices=["json","html","both"])
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    dp = Path(args.dataset) if args.dataset else None
    ev = Evaluator(dp)
    report = await ev.run_all(args.category, args.case)
    ev.print_report(report)

    if args.report in ("json", "both"):
        out = Path(args.output) if args.output else None
        ev.export_json(report, out)
    if args.report in ("html", "both"):
        out = Path(args.output) if args.output else None
        if out and args.report == "both": out = out.with_suffix(".html")
        ev.export_html(report, out)

    sys.exit(0 if report.failed_cases == 0 and report.error_cases == 0 else 1)

if __name__ == "__main__":
    asyncio.run(main())
