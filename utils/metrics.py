"""
智能搜索助手 — Prometheus 指标收集与可用性统计

提供:
  - 请求计数 / 延迟直方图 / 错误率 / Token 用量
  - 滑动窗口可用性统计 (95% SLA 目标)
  - Prometheus 文本格式导出
"""
from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
    REGISTRY,
)

# ── 使用独立 Registry，避免与其它库冲突 ──
_METRICS_REGISTRY: CollectorRegistry = REGISTRY

# ============================================================
# Prometheus 指标定义
# ============================================================

# 请求总数 (按 method / path / status 分组)
http_requests_total = Counter(
    "http_requests_total",
    "HTTP 请求总数",
    ["method", "path", "status_code"],
    registry=_METRICS_REGISTRY,
)

# 请求延迟直方图
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP 请求延迟 (秒)",
    ["method", "path"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=_METRICS_REGISTRY,
)

# SSE 流式请求总数
sse_streams_total = Counter(
    "sse_streams_total",
    "SSE 流式连接总数",
    ["status"],  # "started" | "completed" | "disconnected"
    registry=_METRICS_REGISTRY,
)

# 搜索调用总数
search_requests_total = Counter(
    "search_requests_total",
    "搜索 API 调用总数",
    ["provider", "status"],  # provider=tavily, status=success|error|timeout
    registry=_METRICS_REGISTRY,
)

# Token 用量
tokens_consumed_total = Counter(
    "tokens_consumed_total",
    "LLM Token 消耗总数",
    ["type"],  # "input" | "output"
    registry=_METRICS_REGISTRY,
)

# 活跃会话数
active_sessions_gauge = Gauge(
    "active_sessions",
    "当前活跃会话数",
    registry=_METRICS_REGISTRY,
)

# 搜索延迟直方图
search_duration_seconds = Histogram(
    "search_duration_seconds",
    "搜索调用延迟 (秒)",
    ["provider"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0],
    registry=_METRICS_REGISTRY,
)

# LLM 调用延迟直方图
llm_call_duration_seconds = Histogram(
    "llm_call_duration_seconds",
    "LLM 调用延迟 (秒)",
    ["operation"],  # "rewrite" | "score" | "generate" | "fallback"
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0],
    registry=_METRICS_REGISTRY,
)

# 错误计数器
errors_total = Counter(
    "errors_total",
    "错误总数",
    ["type"],  # "search_failure" | "llm_error" | "timeout" | "fallback_triggered"
    registry=_METRICS_REGISTRY,
)


# ============================================================
# 滑动窗口可用性统计 (95% SLA)
# ============================================================

@dataclass
class _WindowBucket:
    """单个时间桶: 记录该秒内的成功/失败数"""
    timestamp: float
    successes: int = 0
    failures: int = 0


class AvailabilityTracker:
    """
    滑动窗口可用性追踪器。

    按秒级粒度记录每次请求的成功/失败，
    支持查询任意时间窗口内的可用性百分比，
    并可判断是否满足 95% SLA 目标。
    """

    def __init__(self, window_seconds: int = 300) -> None:
        """
        Args:
            window_seconds: 滑动窗口大小 (默认 5 分钟)
        """
        self._window_seconds = window_seconds
        self._buckets: deque[_WindowBucket] = deque()
        self._lock = threading.Lock()

        # ── Prometheus Gauge: 各窗口可用性 ──
        self._availability_gauge = Gauge(
            "http_availability_ratio",
            "HTTP 可用性比率 (滑动窗口)",
            ["window"],
            registry=_METRICS_REGISTRY,
        )

        # ── Prometheus Gauge: 是否低于 95% SLA ──
        self._sla_violation_gauge = Gauge(
            "http_sla_violation",
            "可用性 SLA 违规 (1 = 低于 95% 阈值)",
            ["window"],
            registry=_METRICS_REGISTRY,
        )

    # ── 记录 ──

    def record_success(self) -> None:
        """记录一次成功请求"""
        self._record(1, 0)

    def record_failure(self) -> None:
        """记录一次失败请求"""
        self._record(0, 1)

    def _record(self, successes: int, failures: int) -> None:
        now = time.time()
        with self._lock:
            # 清理过期桶
            cutoff = now - self._window_seconds
            while self._buckets and self._buckets[0].timestamp < cutoff:
                self._buckets.popleft()

            # 如果最后一个桶在同一秒内则合并，否则新建
            current_second = int(now)
            if self._buckets and int(self._buckets[-1].timestamp) == current_second:
                self._buckets[-1].successes += successes
                self._buckets[-1].failures += failures
            else:
                self._buckets.append(
                    _WindowBucket(timestamp=now, successes=successes, failures=failures)
                )

    # ── 查询 ──

    def _compute_availability(self, window_seconds: int) -> tuple[float, int, int]:
        """返回 (ratio, total_successes, total_failures)"""
        now = time.time()
        cutoff = now - window_seconds
        total_s = 0
        total_f = 0
        with self._lock:
            for bucket in self._buckets:
                if bucket.timestamp >= cutoff:
                    total_s += bucket.successes
                    total_f += bucket.failures
        total = total_s + total_f
        ratio = total_s / total if total > 0 else 1.0
        return ratio, total_s, total_f

    def availability_ratio(self, window_seconds: int | None = None) -> float:
        """获取可用性比率 (0.0 ~ 1.0)"""
        w = window_seconds or self._window_seconds
        ratio, _, _ = self._compute_availability(w)
        return ratio

    def meets_sla(self, threshold: float = 0.95) -> bool:
        """当前窗口是否满足 SLA 目标 (默认 95%)"""
        return self.availability_ratio() >= threshold

    @property
    def stats(self) -> dict:
        """返回多窗口统计摘要"""
        windows = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
        }
        result: dict[str, dict] = {}
        for label, secs in windows.items():
            ratio, successes, failures = self._compute_availability(secs)
            total = successes + failures
            result[label] = {
                "availability_pct": round(ratio * 100, 2),
                "successes": successes,
                "failures": failures,
                "total": total,
                "meets_95_sla": ratio >= 0.95,
            }
        return result

    # ── 更新 Prometheus Gauge ──

    def update_prometheus_gauges(self) -> None:
        """将当前可用性数据同步到 Prometheus Gauge"""
        windows = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}
        for label, secs in windows.items():
            ratio, _, _ = self._compute_availability(secs)
            self._availability_gauge.labels(window=label).set(ratio)
            self._sla_violation_gauge.labels(window=label).set(
                1.0 if ratio < 0.95 else 0.0
            )


# ── 全局单例 ──
_availability_tracker: AvailabilityTracker | None = None


def get_availability_tracker() -> AvailabilityTracker:
    global _availability_tracker
    if _availability_tracker is None:
        _availability_tracker = AvailabilityTracker(window_seconds=300)
    return _availability_tracker


# ============================================================
# 便捷记录函数 (供 Agent 流水线调用)
# ============================================================

def record_http_request(
    method: str,
    path: str,
    status_code: int,
    duration: float,
    is_error: bool = False,
) -> None:
    """记录一次 HTTP 请求的指标"""
    http_requests_total.labels(
        method=method, path=path, status_code=str(status_code)
    ).inc()
    http_request_duration_seconds.labels(method=method, path=path).observe(duration)

    tracker = get_availability_tracker()
    if is_error or status_code >= 500:
        tracker.record_failure()
    else:
        tracker.record_success()


def record_sse_stream(status: str) -> None:
    """记录 SSE 流状态 (started / completed / disconnected)"""
    sse_streams_total.labels(status=status).inc()


def record_search(provider: str, status: str, duration: float) -> None:
    """记录搜索调用"""
    search_requests_total.labels(provider=provider, status=status).inc()
    search_duration_seconds.labels(provider=provider).observe(duration)


def record_tokens(input_tokens: int, output_tokens: int) -> None:
    """记录 LLM Token 消耗"""
    if input_tokens > 0:
        tokens_consumed_total.labels(type="input").inc(input_tokens)
    if output_tokens > 0:
        tokens_consumed_total.labels(type="output").inc(output_tokens)


def record_llm_call(operation: str, duration: float) -> None:
    """记录 LLM 调用延迟"""
    llm_call_duration_seconds.labels(operation=operation).observe(duration)


def record_error(error_type: str) -> None:
    """记录错误事件"""
    errors_total.labels(type=error_type).inc()


def update_active_sessions(count: int) -> None:
    """更新活跃会话 Gauge"""
    active_sessions_gauge.set(count)


# ============================================================
# Prometheus 导出
# ============================================================

def get_metrics_text() -> str:
    """生成 Prometheus 文本格式的指标输出"""
    get_availability_tracker().update_prometheus_gauges()
    return generate_latest(_METRICS_REGISTRY).decode("utf-8")


def get_availability_stats() -> dict:
    """返回人类可读的可用性统计 JSON"""
    tracker = get_availability_tracker()
    return tracker.stats


def get_all_stats() -> dict:
    """
    返回完整统计 JSON (包含可用性 + Prometheus 数据快照)。
    用于 /api/metrics?format=json 查询。
    """
    availability = get_availability_stats()

    # 从 Prometheus registry 收集计数器快照
    metrics_snapshot: dict[str, dict] = {}

    for metric_name in [
        "http_requests_total",
        "sse_streams_total",
        "search_requests_total",
        "tokens_consumed_total",
        "errors_total",
    ]:
        metrics_snapshot[metric_name] = {}
        # 遍历 registry 获取样本
        for metric in _METRICS_REGISTRY.collect():
            if metric.name == metric_name:
                for sample in metric.samples:
                    labels = sample.labels
                    # 把标签转成可读的 key
                    label_str = ",".join(
                        f"{k}={v}" for k, v in labels.items() if k != "window"
                    )
                    metrics_snapshot[metric_name][label_str] = sample.value

    return {
        "availability": availability,
        "prometheus_snapshot": metrics_snapshot,
    }
