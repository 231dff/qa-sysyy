"""
智能搜索助手 — 全局配置管理
基于 pydantic-settings，支持 .env 环境变量注入
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 外部 API 配置
# ============================================================
class APIConfig(BaseSettings):
    """第三方 API 密钥与端点"""

    # LLM API (兼容 OpenAI 协议)
    llm_api_key: SecretStr = Field(
        default=SecretStr("sk-your-api-key"),
        description="LLM API Key",
    )
    llm_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="LLM API Base URL",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="默认 LLM 模型名称",
    )
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="LLM 推理温度",
    )

    # Tavily 搜索 API
    tavily_api_key: SecretStr = Field(
        default=SecretStr("tvly-your-api-key"),
        description="Tavily Search API Key",
    )
    tavily_api_base: str = Field(
        default="https://api.tavily.com",
        description="Tavily API Base URL",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ============================================================
# Agent 行为配置
# ============================================================
class AgentConfig:
    """Agent 运行时参数"""

    # ── 搜索 ──
    max_search_results: int = 10           # Tavily 单次最大返回条数
    search_depth: str = "advanced"         # "basic" | "advanced"
    search_timeout: float = 15.0           # 搜索 API 超时 (秒)
    include_domains: list[str] = []        # 白名单域名 (空 = 不限)
    exclude_domains: list[str] = []        # 黑名单域名

    # ── 上下文裁剪 ──
    top_k_fragments: int = 5               # 相关性打分后保留的 Top-K 片段
    max_context_tokens: int = 3000         # 单轮最大上下文 token 数
    dedup_window: int = 50                 # URL 去重滑动窗口大小

    # ── 会话 ──
    max_history_turns: int = 10            # 最多保留的历史对话轮次
    session_ttl_seconds: int = 3600        # 会话缓存过期时间 (秒)

    # ── 重试 ──
    max_retries: int = 3                   # 最大重试次数
    base_delay: float = 1.0                # 指数退避基础延迟 (秒)
    max_delay: float = 30.0                # 指数退避最大延迟 (秒)

    # ── 限流 ──
    max_requests_per_minute: int = 30      # 每分钟最大请求数
    max_searches_per_minute: int = 10      # 每分钟最大搜索次数

    # ── 调试 ──
    verbose: bool = False                  # 是否输出详细日志


# ============================================================
# 单例配置
# ============================================================
_api_config: Optional[APIConfig] = None
_agent_config: Optional[AgentConfig] = None


def get_api_config() -> APIConfig:
    """获取 API 配置单例"""
    global _api_config
    if _api_config is None:
        _api_config = APIConfig()
    return _api_config


def get_agent_config() -> AgentConfig:
    """获取 Agent 配置单例"""
    global _agent_config
    if _agent_config is None:
        _agent_config = AgentConfig()
    return _agent_config
