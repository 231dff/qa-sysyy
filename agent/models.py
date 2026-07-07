"""
智能搜索助手 — 数据模型定义
所有 Agent 状态、工具输入/输出均使用 Pydantic 强类型约束
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, Field


# ============================================================
# 搜索相关
# ============================================================
class SearchResult(BaseModel):
    """单条搜索结果"""
    url: str = Field(..., description="结果 URL")
    title: str = Field(default="", description="标题")
    content: str = Field(default="", description="摘要/正文片段")
    score: float = Field(default=0.0, description="原始搜索相关性分数")
    raw_response: Optional[dict[str, Any]] = Field(
        default=None, description="API 原始响应 (调试用)", repr=False
    )


class SearchResponse(BaseModel):
    """搜索 API 返回的批量结果"""
    query: str = Field(..., description="实际搜索查询词")
    results: list[SearchResult] = Field(default_factory=list)
    total_count: int = Field(default=0)
    search_time_ms: float = Field(default=0.0)
    error: Optional[str] = Field(default=None)


class RelevanceScore(BaseModel):
    """LLM 对单条结果的相关性打分"""
    url: str
    title: str
    relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="与用户原始问题的相关性 (0-1)",
    )
    reason: str = Field(default="", description="评分理由 (简短)")


class RelevanceScores(BaseModel):
    """批量相关性打分"""
    results: list[RelevanceScore] = Field(default_factory=list)


# ============================================================
# 查询改写
# ============================================================
class RewrittenQuery(BaseModel):
    """查询改写结果"""
    original: str = Field(..., description="用户原始输入")
    rewritten: str = Field(..., description="改写后的搜索查询词")
    language: str = Field(default="zh", description="检测到的语言")
    intent: str = Field(
        default="factual",
        description="意图类型: factual | opinion | news | guide | comparison",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="可选的子查询拆分 (复杂问题)",
    )


# ============================================================
# 答案生成
# ============================================================
class Source(BaseModel):
    """答案引用的信息源"""
    url: str
    title: str
    snippet: str = Field(default="", description="引用的片段")


class GeneratedAnswer(BaseModel):
    """最终生成的答案"""
    query: str = Field(..., description="原始用户问题")
    answer: str = Field(..., description="最终答案 (Markdown)")
    sources: list[Source] = Field(default_factory=list)
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="答案置信度",
    )
    is_fallback: bool = Field(
        default=False,
        description="是否来自降级回答 (LLM 知识)",
    )
    tokens_used: int = Field(default=0)
    latency_ms: float = Field(default=0.0)


# ============================================================
# Agent 状态图 (StateGraph TypedDict)
# ============================================================
class AgentState(TypedDict, total=False):
    """LangGraph Agent 全局状态"""
    # 会话标识
    session_id: str

    # 用户输入
    user_query: str
    user_query_raw: str                          # 原始未处理输入

    # 查询改写
    rewritten_queries: list[RewrittenQuery]

    # 搜索
    search_results: list[SearchResult]           # 原始搜索结果 (去重前)
    deduped_results: list[SearchResult]          # URL 去重后

    # 相关性打分
    relevance_scores: RelevanceScores            # 所有结果的打分
    top_results: list[SearchResult]              # Top-K 过滤后

    # 答案
    final_answer: GeneratedAnswer

    # 对话历史
    history: list[dict[str, str]]                # [{"role":"user","content":...}, ...]

    # 元信息
    error: Optional[str]
    retry_count: int
    fallback_triggered: bool
    search_failed: bool

    # 计时
    started_at: str                              # ISO 时间戳
    completed_at: str                            # ISO 时间戳


# ============================================================
# 工具元数据
# ============================================================
class ToolCategory(str, Enum):
    """工具分类"""
    SEARCH = "search"
    REWRITE = "rewrite"
    RELEVANCE = "relevance"
    FALLBACK = "fallback"


@dataclass
class ToolMetadata:
    """工具注册元信息 — 职责单一"""
    name: str
    category: ToolCategory
    description: str
    version: str = "1.0.0"


# ============================================================
# 会话与缓存
# ============================================================
@dataclass
class SessionEntry:
    """会话缓存条目"""
    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    last_query: str = ""
    created_at: float = field(default=0.0)
    last_access: float = field(default=0.0)

    def touch(self, now: float) -> None:
        self.last_access = now


# ============================================================
# 限流
# ============================================================
@dataclass
class RateLimitState:
    """滑动窗口限流状态"""
    request_timestamps: list[float] = field(default_factory=list)
    search_timestamps: list[float] = field(default_factory=list)
