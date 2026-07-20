"""
基准测试: 量化链路延迟与 Token 开销

对比两种模式 (均 mock 外部 API，仅测量本地处理开销):
  Mode A (优化前): 串行子查询 + LLM 打分
  Mode B (优化后): 并行子查询 + Tavily 原始分

用法:
    python -m evals.benchmark                # 仅摘要
    python -m evals.benchmark --json          # 输出 JSON
    python -m evals.benchmark --html          # 输出 HTML 报告
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
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import SearchAgent, node_generate_answer, node_rewrite_query, node_search
from agent.models import AgentState, GeneratedAnswer, RewrittenQuery, SearchResult
from config import get_agent_config, get_api_config
from tools.registry import (
    rewrite_query,
    search_and_filter_pipeline,
    score_relevance,
    tavily_search,
)
from utils.helpers import count_tokens, trim_context, URLDeduplicator
from utils.http_client import get_http_client


# ═══════════════════════════════════════════════════════════
# 模拟数据
# ═══════════════════════════════════════════════════════════
MOCK_TAVILY_RESULTS = [
    {"url": f"https://example.com/result-{i}",
     "title": f"标题-{i}",
     "content": f"这是第{i}条搜索结果的正文内容。" + "包含一些背景信息和上下文。" * 20,
     "score": round(0.95 - i * 0.08, 2)}
    for i in range(1, 11)  # 10 条结果, 分从 0.87→0.15
]

REWRITE_OUT = json.dumps({
    "rewritten": "2025年7月 重大新闻",
    "language": "zh",
    "intent": "news",
    "sub_queries": ["中国 2025 新闻", "国际 2025 新闻"],
})

SCORE_OUT = json.dumps({
    "results": [{"url": f"https://example.com/result-{i}", "title": f"标题-{i}",
                 "relevance": round(max(0.4, 0.95 - i * 0.1), 2),
                 "reason": "相关"} for i in range(1, 11)]
})

ANSWER_OUT = "## 今日要闻\n\n根据最新搜索结果:\n\n1. **AI 领域** 有新突破 [1]\n2. **科技** 持续发展 [2]\n\n📚 参考来源"


@dataclass
class StepMetrics:
    label: str
    duration_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0
    note: str = ""


@dataclass
class RunResult:
    mode: str  # "优化前 (串行 + LLM 打分)" | "优化后 (并行 + Tavily 分)"
    total_duration_ms: float
    steps: list[StepMetrics] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_llm_calls: int = 0
    error: Optional[str] = None

    @property
    def first_token_ms(self) -> float:
        """估计首 token 延迟: 到达流式生成阶段前的耗时"""
        pre_generate = 0.0
        for s in self.steps:
            if "生成" in s.label:
                break
            pre_generate += s.duration_ms
        return pre_generate


# ═══════════════════════════════════════════════════════════
# Mock 工厂
# ═══════════════════════════════════════════════════════════
def _nop_resp(json_data: dict, status_code: int = 200):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_data
    r.raise_for_status = MagicMock()
    return r


def _llm_resp(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _make_handler(calls_log: list, mode: str):
    """返回 mock post handler，记录每次 LLM 调用的 prompt token 数"""
    async def handler(*args, **kwargs):
        body = kwargs.get("json", {})
        msgs = body.get("messages", [])
        # 计算 prompt tokens
        sys_text = msgs[0]["content"] if msgs else ""
        user_text = msgs[-1]["content"] if len(msgs) > 1 else ""
        prompt_tok = count_tokens(sys_text + "\n" + user_text)

        calls_log.append({
            "system_fragment": sys_text[:40],
            "prompt_tokens": prompt_tok,
            "stream": body.get("stream", False),
        })

        if "搜索查询优化专家" in sys_text:
            return _nop_resp(_llm_resp(REWRITE_OUT))
        elif "搜索结果质量评估专家" in sys_text:
            return _nop_resp(_llm_resp(SCORE_OUT))
        elif "实时信息问答助手" in sys_text or "智能问答助手" in sys_text:
            if body.get("stream"):
                # 流式: mock stream response
                return _nop_resp({})  # handled differently
            return _nop_resp(_llm_resp(ANSWER_OUT))
        else:
            # Tavily
            return _nop_resp({"results": MOCK_TAVILY_RESULTS})
    return handler


class _Patcher:
    """patch get_http_client 注入 mock"""
    def __init__(self, handler):
        self.handler = handler
        self._patches = []

    def __enter__(self):
        from contextlib import asynccontextmanager

        m = MagicMock()
        m.post = AsyncMock(side_effect=self.handler)

        # stream 支持
        async def _mock_aiter_lines():
            for token in ANSWER_OUT.split(" "):
                chunk = json.dumps({
                    "choices": [{"delta": {"content": token + " "}}]
                })
                yield f"data: {chunk}"
            yield "data: [DONE]"

        mock_stream_resp = MagicMock()
        mock_stream_resp.aiter_lines = _mock_aiter_lines
        mock_stream_resp.raise_for_status = MagicMock()
        m.stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream_resp)

        @asynccontextmanager
        async def _mock_get_http_client(timeout=30.0):
            yield m

        for target in ["tools.registry.get_http_client", "agent.graph.get_http_client"]:
            p = patch(target, side_effect=_mock_get_http_client)
            p.start()
            self._patches.append(p)
        return self

    def __exit__(self, *args):
        for p in self._patches:
            p.stop()


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════
async def run_mode_a() -> RunResult:
    """Mode A: 优化前 — 串行子查询 + LLM 打分"""
    calls_log = []
    handler = _make_handler(calls_log, "A")
    steps: list[StepMetrics] = []
    cfg = get_agent_config()

    with _Patcher(handler):
        # ---- Step 1: 查询改写 ----
        t0 = time.perf_counter()
        rw = await rewrite_query("今天有什么重大新闻？", [])
        t1 = time.perf_counter()
        rewrite_call = [c for c in calls_log if "搜索查询优化专家" in c["system_fragment"]][-1]
        steps.append(StepMetrics("1. 查询改写 (LLM)", (t1 - t0) * 1000,
                                 prompt_tokens=rewrite_call["prompt_tokens"],
                                 completion_tokens=count_tokens(REWRITE_OUT),
                                 call_count=1))

        # ---- Step 2: 搜索 + 打分 (串行, 含 LLM 打分) ----
        queries = [rw.rewritten] + rw.sub_queries[:2]
        dedup = URLDeduplicator(window_size=cfg.dedup_window)
        all_deduped = []
        all_fragments = []

        search_start = time.perf_counter()
        scoring_calls = 0
        scoring_prompt_tok = 0
        calls_before = len(calls_log)

        for q in queries:
            t_q0 = time.perf_counter()
            # 强制走 LLM 打分
            deduped_res, fragments = await search_and_filter_pipeline(
                q, deduplicator=dedup, use_llm_scoring=True
            )
            t_q1 = time.perf_counter()
            all_deduped.extend(deduped_res)
            all_fragments.extend(fragments)

            # 统计这一步的打分调用
            new_calls = calls_log[calls_before:]
            sc_calls = [c for c in new_calls if "搜索结果质量评估专家" in c["system_fragment"]]
            scoring_calls += len(sc_calls)
            scoring_prompt_tok += sum(c["prompt_tokens"] for c in sc_calls)
            calls_before = len(calls_log)

        search_elapsed = (time.perf_counter() - search_start) * 1000
        steps.append(StepMetrics("2. 搜索+打分 (串行×3次)",
                                 search_elapsed,
                                 prompt_tokens=scoring_prompt_tok,
                                 call_count=scoring_calls,
                                 note=f"Tavily 3次 + LLM打分 3次, 有效结果 {len(all_deduped)}条"))

        # ---- Step 3: 答案生成 ----
        gen_start = time.perf_counter()
        context = "\n".join(f"{url}: {content}" for url, content, _ in all_fragments)
        gen_prompt = f"用户问题: 今天有什么重大新闻？\n搜索结果:\n{context}"
        calls_before = len(calls_log)
        # 直接调 mock handler (避免真实 httpx 请求)
        await handler(*(MagicMock(),), **{"json": {
            "model": "mock",
            "messages": [{"role": "system", "content": "实时信息问答助手"},
                         {"role": "user", "content": gen_prompt}],
        }})
        gen_elapsed = (time.perf_counter() - gen_start) * 1000
        gen_calls = [c for c in calls_log[calls_before:] if "实时信息问答助手" in c["system_fragment"]]
        gen_prompt_tok = sum(c["prompt_tokens"] for c in gen_calls)
        steps.append(StepMetrics("3. 答案生成 (LLM)", gen_elapsed,
                                 prompt_tokens=gen_prompt_tok,
                                 call_count=len(gen_calls),
                                 note="阻塞路径"))

    total = sum(s.duration_ms for s in steps)
    return RunResult(
        mode="Mode A: 优化前 (串行子查询 + LLM打分)",
        total_duration_ms=total,
        steps=steps,
        total_prompt_tokens=sum(s.prompt_tokens for s in steps),
        total_completion_tokens=sum(s.completion_tokens for s in steps),
        total_llm_calls=sum(s.call_count for s in steps),
    )


async def run_mode_b(streaming: bool = False) -> RunResult:
    """Mode B: 优化后 — 并行子查询 + 跳过 LLM 打分"""
    calls_log = []
    handler = _make_handler(calls_log, "B")
    steps: list[StepMetrics] = []
    cfg = get_agent_config()

    with _Patcher(handler):
        # ---- Step 1: 查询改写 ----
        t0 = time.perf_counter()
        rw = await rewrite_query("今天有什么重大新闻？", [])
        t1 = time.perf_counter()
        rewrite_call = [c for c in calls_log if "搜索查询优化专家" in c["system_fragment"]][-1]
        steps.append(StepMetrics("1. 查询改写 (LLM)", (t1 - t0) * 1000,
                                 prompt_tokens=rewrite_call["prompt_tokens"],
                                 completion_tokens=count_tokens(REWRITE_OUT),
                                 call_count=1))

        # ---- Step 2: 搜索 (并行, 无 LLM 打分) ----
        queries = [rw.rewritten] + rw.sub_queries[:2]
        dedup = URLDeduplicator(window_size=cfg.dedup_window)
        all_deduped = []
        all_fragments = []

        search_start = time.perf_counter()

        async def _search_one(q: str):
            return await search_and_filter_pipeline(
                q, deduplicator=dedup, use_llm_scoring=False
            )

        results = await asyncio.gather(*[_search_one(q) for q in queries])
        for deduped_res, fragments in results:
            all_deduped.extend(deduped_res)
            all_fragments.extend(fragments)

        search_elapsed = (time.perf_counter() - search_start) * 1000
        # 并行后总时间约等于最慢那路
        steps.append(StepMetrics("2. 搜索 (并行×3, 无LLM打分)",
                                 search_elapsed,
                                 prompt_tokens=0,
                                 call_count=0,
                                 note=f"3路 Tavily 并行, 无LLM打分, 有效结果 {len(all_deduped)}条"))

        # ---- Step 3: 答案生成 ----
        gen_start = time.perf_counter()
        context = "\n".join(f"{url}: {content}" for url, content, _ in all_fragments)
        gen_prompt = f"用户问题: 今天有什么重大新闻？\n搜索结果:\n{context}"
        calls_before = len(calls_log)
        # 直接调 mock handler (避免真实 httpx 请求)
        await handler(*(MagicMock(),), **{"json": {
            "model": "mock",
            "messages": [{"role": "system", "content": "实时信息问答助手"},
                         {"role": "user", "content": gen_prompt}],
        }})
        gen_elapsed = (time.perf_counter() - gen_start) * 1000
        gen_calls = [c for c in calls_log[calls_before:] if "实时信息问答助手" in c["system_fragment"]]
        gen_prompt_tok = sum(c["prompt_tokens"] for c in gen_calls)
        steps.append(StepMetrics("3. 答案生成 (LLM)", gen_elapsed,
                                 prompt_tokens=gen_prompt_tok,
                                 call_count=len(gen_calls),
                                 note="阻塞路径"))

    total = sum(s.duration_ms for s in steps)
    return RunResult(
        mode="Mode B: 优化后 (并行搜索 + Tavily原始分)",
        total_duration_ms=total,
        steps=steps,
        total_prompt_tokens=sum(s.prompt_tokens for s in steps),
        total_completion_tokens=sum(s.completion_tokens for s in steps),
        total_llm_calls=sum(s.call_count for s in steps),
    )


# ═══════════════════════════════════════════════════════════
# 额外: Token 消耗拆解
# ═══════════════════════════════════════════════════════════
@dataclass
class TokenBreakdown:
    component: str
    input_tokens: int
    output_tokens: int
    note: str = ""

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


def analyze_token_breakdown() -> list[TokenBreakdown]:
    """静态分析各环节的 token 开销 (通过直接计算 prompt 模板)."""
    from utils.helpers import count_tokens
    from tools.registry import (
        REWRITE_SYSTEM_PROMPT,
        RELEVANCE_SYSTEM_PROMPT,
        FALLBACK_SYSTEM_PROMPT,
    )
    from agent.graph import ANSWER_SYSTEM_PROMPT

    results = []

    # 1. 查询改写
    rewrite_user = "对话历史:\n(无)\n\n当前用户问题:\n今天有什么重大新闻？\n\n请输出 JSON:"
    rewrite_in = count_tokens(REWRITE_SYSTEM_PROMPT + rewrite_user)
    rewrite_out = count_tokens(REWRITE_OUT)
    results.append(TokenBreakdown("查询改写 (LLM)", rewrite_in, rewrite_out,
                                  "每次都执行，无法省略"))

    # 2. 相关性打分
    # 构造场景: 10条结果，每条 content 截断为 500 字符
    scoring_candidates = json.dumps([
        {"index": i, "url": f"https://example.com/{i}",
         "title": f"标题-{i}",
         "content": f"内容文本-{i} " * 50}  # ~500 chars
        for i in range(10)
    ], ensure_ascii=False, indent=2)
    scoring_user = f"用户问题:\n今天有什么重大新闻？\n\n搜索结果候选:\n{scoring_candidates}\n\n请为每条结果打分:"
    scoring_in = count_tokens(RELEVANCE_SYSTEM_PROMPT + scoring_user)
    scoring_out = count_tokens(SCORE_OUT)
    results.append(TokenBreakdown("相关性打分 (LLM)", scoring_in, scoring_out,
                                  "默认跳过改用 Tavily 分 → 每次搜索省此开销"))

    # 3. 答案生成
    answer_user = "用户问题: 今天有什么重大新闻？\n实时搜索结果: (5条裁剪后片段)\n请基于以上搜索结果回答问题:"
    answer_in = count_tokens(ANSWER_SYSTEM_PROMPT + answer_user)
    results.append(TokenBreakdown("答案生成 (LLM)", answer_in, count_tokens(ANSWER_OUT),
                                  "必须执行，无法省略"))

    # 4. 降级回答 (仅在搜索失败时触发)
    fallback_user = "今天有什么重大新闻？"
    fallback_in = count_tokens(FALLBACK_SYSTEM_PROMPT + fallback_user)
    results.append(TokenBreakdown("降级回答 (LLM)", fallback_in, count_tokens(ANSWER_OUT[:200]),
                                  "仅搜索失败时触发，不在正常流程中"))

    return results


# ═══════════════════════════════════════════════════════════
# 输出
# ═══════════════════════════════════════════════════════════
def print_report(result_a: RunResult, result_b: RunResult, token_bd: list[TokenBreakdown]):
    """打印对比报告到控制台"""
    sep = "=" * 70

    print(f"\n{sep}")
    print("  基准测试报告 — 延迟 & Token 对比")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(sep)

    # ---- 延迟对比 ----
    print(f"\n{'─' * 70}")
    print("  一、端到端延迟对比 (mock API, 仅计算本地编排开销)")
    print(f"{'─' * 70}")

    for result in [result_a, result_b]:
        print(f"\n  【{result.mode}】")
        print(f"  总耗时: {result.total_duration_ms:.1f}ms  |  LLM 调用: {result.total_llm_calls}次")
        for s in result.steps:
            bar = "█" * max(1, int(s.duration_ms / max(result.total_duration_ms, 1) * 30))
            print(f"  {s.label:<30s} {s.duration_ms:8.1f}ms  {bar}")
            if s.note:
                print(f"  {'':30s} ({s.note})")

    # 关键指标
    delta_ms = result_a.total_duration_ms - result_b.total_duration_ms
    reduction = (delta_ms / result_a.total_duration_ms * 100) if result_a.total_duration_ms else 0
    delta_calls = result_a.total_llm_calls - result_b.total_llm_calls
    delta_tok = result_a.total_prompt_tokens - result_b.total_prompt_tokens
    tok_reduction = (delta_tok / result_a.total_prompt_tokens * 100) if result_a.total_prompt_tokens else 0

    print(f"\n  {'─' * 50}")
    print(f"  📊 关键结论 (端到端)")
    print(f"  {'─' * 50}")
    print(f"  延迟变化:    {result_a.total_duration_ms:.0f}ms → {result_b.total_duration_ms:.0f}ms  "
          f"(节省 {delta_ms:.0f}ms / {reduction:.0f}%)")
    print(f"  LLM 调用:    {result_a.total_llm_calls}次 → {result_b.total_llm_calls}次  "
          f"(减少 {delta_calls}次)")
    print(f"  Prompt Token: {result_a.total_prompt_tokens} → {result_b.total_prompt_tokens}  "
          f"(节省 {delta_tok} tokens / {tok_reduction:.0f}%)")
    print(f"  首 token 估算: {result_a.first_token_ms:.0f}ms → {result_b.first_token_ms:.0f}ms  "
          f"(节省 {result_a.first_token_ms - result_b.first_token_ms:.0f}ms)")

    # ---- Token 拆解 ----
    print(f"\n{'─' * 70}")
    print("  二、Token 消耗拆解 (按环节)")
    print(f"{'─' * 70}\n")
    print(f"  {'环节':<20s} {'输入':>8s} {'输出':>8s} {'合计':>8s}  备注")
    print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8}  {'─' * 30}")

    total_avoidable_in = 0
    for bd in token_bd:
        tag = "★" if "无法省略" not in bd.note else " "
        print(f"{tag} {bd.component:<18s} {bd.input_tokens:>8,} {bd.output_tokens:>8,} {bd.total:>8,}  {bd.note}")
        if "跳过" in bd.note or "故障" in bd.note:
            total_avoidable_in += bd.input_tokens

    grand_total_in = sum(bd.input_tokens for bd in token_bd)
    print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8}")
    print(f"  {'合计':<20s} {grand_total_in:>8,}")
    print(f"\n  可省 prompt tokens (打分环节): {total_avoidable_in:,}")
    print(f"  占总 prompt: {total_avoidable_in / grand_total_in * 100:.0f}%")

    # ---- 每轮对话总开销 ----
    print(f"\n{'─' * 70}")
    print("  三、单次问答总 Token 开销对比")
    print(f"{'─' * 70}\n")
    print(f"  | 模式       | LLM 调用 | Prompt Tokens | 说明                    |")
    print(f"  |{'─' * 12}|{'─' * 10}|{'─' * 15}|{'─' * 25}|")
    # Mode A: 改写 + 3×打分 + 生成 = 5 次 LLM 调用
    a_tok = token_bd[0].input_tokens + token_bd[1].input_tokens * 3 + token_bd[2].input_tokens
    print(f"  | 优化前     |    5次   | {a_tok:>13,} | 改写+打分×3+生成         |")
    # Mode B: 改写 + 生成 = 2 次
    b_tok = token_bd[0].input_tokens + token_bd[2].input_tokens
    print(f"  | 优化后     |    2次   | {b_tok:>13,} | 改写+生成                |")
    print(f"  | 节省       |    3次   | {a_tok - b_tok:>13,} | ({((a_tok - b_tok) / a_tok * 100):.0f}%)                    |")

    print(f"\n{sep}")
    print("  注意: 以上数字为 prompt 模板静态计算 + mock 编排耗时")
    print("  真实延迟取决于 LLM API 和 Tavily API 的网络 RTT")
    print(sep)


def export_json(result_a: RunResult, result_b: RunResult, token_bd: list[TokenBreakdown],
                path: Optional[Path] = None):
    path = path or PROJECT_ROOT / "evals" / "benchmark.json"
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode_a": {
            "label": result_a.mode,
            "total_duration_ms": round(result_a.total_duration_ms, 1),
            "first_token_ms": round(result_a.first_token_ms, 1),
            "llm_calls": result_a.total_llm_calls,
            "prompt_tokens": result_a.total_prompt_tokens,
            "steps": [{"label": s.label, "duration_ms": round(s.duration_ms, 1),
                       "prompt_tokens": s.prompt_tokens, "llm_calls": s.call_count, "note": s.note}
                      for s in result_a.steps],
        },
        "mode_b": {
            "label": result_b.mode,
            "total_duration_ms": round(result_b.total_duration_ms, 1),
            "first_token_ms": round(result_b.first_token_ms, 1),
            "llm_calls": result_b.total_llm_calls,
            "prompt_tokens": result_b.total_prompt_tokens,
            "steps": [{"label": s.label, "duration_ms": round(s.duration_ms, 1),
                       "prompt_tokens": s.prompt_tokens, "llm_calls": s.call_count, "note": s.note}
                      for s in result_b.steps],
        },
        "comparison": {
            "latency_saved_ms": round(result_a.total_duration_ms - result_b.total_duration_ms, 1),
            "latency_reduction_pct": round(
                (result_a.total_duration_ms - result_b.total_duration_ms) / max(result_a.total_duration_ms, 1) * 100, 1
            ),
            "llm_calls_reduced": result_a.total_llm_calls - result_b.total_llm_calls,
            "prompt_tokens_saved": result_a.total_prompt_tokens - result_b.total_prompt_tokens,
        },
        "token_breakdown": [{"component": bd.component, "input_tokens": bd.input_tokens,
                             "output_tokens": bd.output_tokens, "note": bd.note} for bd in token_bd],
    }
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[json] {path}")


def export_html(result_a: RunResult, result_b: RunResult, token_bd: list[TokenBreakdown],
                path: Optional[Path] = None):
    path = path or PROJECT_ROOT / "evals" / "benchmark.html"
    delta_ms = result_a.total_duration_ms - result_b.total_duration_ms
    reduction = (delta_ms / result_a.total_duration_ms * 100) if result_a.total_duration_ms else 0
    delta_tok = result_a.total_prompt_tokens - result_b.total_prompt_tokens
    tok_reduction = (delta_tok / result_a.total_prompt_tokens * 100) if result_a.total_prompt_tokens else 0

    def _step_rows(result: RunResult) -> str:
        rows = ""
        for s in result.steps:
            rows += f"<tr><td>{s.label}</td><td>{s.duration_ms:.0f}</td><td>{s.prompt_tokens:,}</td><td>{s.call_count}</td><td>{s.note or ''}</td></tr>"
        return rows

    token_rows = ""
    for bd in token_bd:
        token_rows += f"<tr><td>{bd.component}</td><td>{bd.input_tokens:,}</td><td>{bd.output_tokens:,}</td><td>{bd.total:,}</td><td>{bd.note}</td></tr>"

    a_tok = token_bd[0].input_tokens + token_bd[1].input_tokens * 3 + token_bd[2].input_tokens
    b_tok = token_bd[0].input_tokens + token_bd[2].input_tokens

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><title>基准测试报告</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:system-ui,sans-serif;background:#f8fafc;padding:2rem}}
.container{{max-width:1000px;margin:0 auto}}h1{{font-size:1.5rem;margin-bottom:.5rem}}h3{{margin:1rem 0 .5rem}}
.card{{background:#fff;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;box-shadow:0 1px 3px #0001}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem}}
.stat{{text-align:center;padding:1rem;background:#f1f5f9;border-radius:8px}}
.stat .v{{font-size:1.8rem;font-weight:700}}.stat .l{{font-size:.75rem;color:#64748b}}
.positive{{color:#16a34a}}.negative{{color:#dc2626}}
table{{width:100%;border-collapse:collapse;margin-top:.5rem}}th,td{{text-align:left;padding:.5rem .8rem;border-bottom:1px solid #e2e8f0}}
th{{background:#f8fafc;font-weight:600}}tr:hover{{background:#f8fafc}}
.bar-container{{display:flex;gap:.5rem;align-items:center;margin:.5rem 0}}
.bar-a,.bar-b{{height:24px;border-radius:4px;display:flex;align-items:center;padding-left:8px;font-size:.75rem;color:#fff;font-weight:600}}
.bar-a{{background:#dc2626}}.bar-b{{background:#16a34a}}
.footer{{text-align:center;color:#94a3b8;font-size:.8rem;margin-top:2rem}}
</style></head><body><div class="container">
<h1>📊 基准测试报告</h1>
<p>v1.0 &middot; {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} &middot; mock 外部 API</p>

<div class="card"><h3>⏱️ 延迟对比</h3>
<div class="grid">
<div class="stat"><div class="v">{result_a.total_duration_ms:.0f}<span style="font-size:.9rem">ms</span></div><div class="l">优化前总耗时</div></div>
<div class="stat"><div class="v positive">{result_b.total_duration_ms:.0f}<span style="font-size:.9rem">ms</span></div><div class="l">优化后总耗时</div></div>
<div class="stat"><div class="v positive">-{delta_ms:.0f}<span style="font-size:.9rem">ms</span></div><div class="l">节省</div></div>
<div class="stat"><div class="v positive">{reduction:.0f}%</div><div class="l">延迟降低</div></div>
</div>

<div style="margin-top:1rem">
<h4>Mode A (优化前)</h4>
<table><tr><th>环节</th><th>耗时(ms)</th><th>Prompt Token</th><th>LLM调用</th><th>备注</th></tr>{_step_rows(result_a)}</table>
<h4>Mode B (优化后)</h4>
<table><tr><th>环节</th><th>耗时(ms)</th><th>Prompt Token</th><th>LLM调用</th><th>备注</th></tr>{_step_rows(result_b)}</table>
</div></div>

<div class="card"><h3>🧮 Token 对比</h3>
<div class="grid">
<div class="stat"><div class="v">{result_a.total_llm_calls}<span style="font-size:.9rem">次</span></div><div class="l">优化前 LLM 调用</div></div>
<div class="stat"><div class="v positive">{result_b.total_llm_calls}<span style="font-size:.9rem">次</span></div><div class="l">优化后 LLM 调用</div></div>
<div class="stat"><div class="v">{result_a.total_prompt_tokens:,}</div><div class="l">优化前 Prompt Token</div></div>
<div class="stat"><div class="v positive">{result_b.total_prompt_tokens:,} <span style="font-size:.75rem">({tok_reduction:.0f}%↓)</span></div><div class="l">优化后 Prompt Token</div></div>
</div></div>

<div class="card"><h3>🔍 Token 消耗拆解</h3>
<table><tr><th>环节</th><th>输入 Token</th><th>输出 Token</th><th>合计</th><th>备注</th></tr>{token_rows}</table>
</div>

<div class="card"><h3>📋 单次问答总开销</h3>
<table><tr><th>模式</th><th>LLM 调用</th><th>Prompt Tokens</th><th>说明</th></tr>
<tr><td>优化前</td><td>5次</td><td>{a_tok:,}</td><td>改写 + 打分×3 + 生成</td></tr>
<tr><td>优化后</td><td>2次</td><td>{b_tok:,}</td><td>改写 + 生成</td></tr>
<tr style="font-weight:700;color:#16a34a"><td>节省</td><td>3次</td><td>{a_tok - b_tok:,} ({(a_tok - b_tok) / a_tok * 100:.0f}%)</td><td></td></tr>
</table></div>

<div class="footer">Mock基准测试 &middot; 真实延迟取决于 API RTT &middot; {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</div>
</div></body></html>"""
    open(path, "w", encoding="utf-8").write(html)
    print(f"[html] {path}")


# ═══════════════════════════════════════════════════════════
async def main():
    ap = argparse.ArgumentParser(description="基准测试: 延迟与 Token 对比")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    print("\n[bench] 运行基准测试...\n")

    result_a = await run_mode_a()
    result_b = await run_mode_b()
    token_bd = analyze_token_breakdown()

    print_report(result_a, result_b, token_bd)

    out = Path(args.output) if args.output else None
    if args.json:
        export_json(result_a, result_b, token_bd, out)
    if args.html:
        export_html(result_a, result_b, token_bd, out)

    # 默认都输出
    if not args.json and not args.html:
        export_json(result_a, result_b, token_bd, out)
        export_html(result_a, result_b, token_bd, out)


if __name__ == "__main__":
    asyncio.run(main())
