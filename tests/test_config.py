"""
智能搜索助手 — 配置加载测试
测试范围: APIConfig (环境变量), AgentConfig (默认值与可变性), 单例函数
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from config import (
    PROJECT_ROOT,
    AgentConfig,
    get_agent_config,
    get_api_config,
)


# ============================================================
# 项目根目录测试
# ============================================================
class TestProjectRoot:
    """PROJECT_ROOT 路径验证"""

    def test_project_root_exists(self):
        """项目根目录应存在"""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_project_root_contains_key_files(self):
        """根目录应包含核心文件"""
        assert (PROJECT_ROOT / "config.py").exists()
        assert (PROJECT_ROOT / "app.py").exists()
        assert (PROJECT_ROOT / "agent" / "graph.py").exists()


# ============================================================
# APIConfig 测试
# ============================================================
class TestAPIConfig:
    """API 配置 (pydantic-settings) 测试"""

    def test_default_values(self):
        """默认值检查 — pydantic-settings 定义的默认值"""
        from config import APIConfig
        # 创建不使用 .env 的实例来测试真正的默认值
        cfg = APIConfig(_env_file=None)

        assert cfg.llm_api_base == "https://api.openai.com/v1"
        assert cfg.llm_model == "gpt-4o-mini"
        assert cfg.llm_temperature == 0.3
        assert cfg.tavily_api_base == "https://api.tavily.com"

    def test_secret_str_masking(self):
        """SecretStr 不应在 repr/str 中暴露真实值"""
        cfg = get_api_config()

        # SecretStr 的 repr 应隐藏密钥
        secret_repr = repr(cfg.llm_api_key)
        assert "sk-your-api-key" not in secret_repr
        assert "SecretStr" in secret_repr or "**********" in secret_repr

    def test_get_secret_value(self):
        """get_secret_value() 返回原始值"""
        # .env 可能覆盖默认值，只检查返回非空字符串
        cfg = get_api_config()
        raw = cfg.llm_api_key.get_secret_value()
        assert isinstance(raw, str) and len(raw) > 0

    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variable_loading(self):
        """环境变量应覆盖默认值"""
        # 需要重新导入以刷新配置单例
        with patch.dict(os.environ, {
            "LLM_API_KEY": "sk-custom-key",
            "LLM_MODEL": "gpt-4o",
            "LLM_TEMPERATURE": "0.7",
            "TAVILY_API_KEY": "tvly-custom",
        }):
            # 由于单例缓存，需要手动创建新实例
            from config import APIConfig
            cfg = APIConfig()

            assert cfg.llm_model == "gpt-4o"
            assert cfg.llm_temperature == 0.7
            assert cfg.llm_api_key.get_secret_value() == "sk-custom-key"
            assert cfg.tavily_api_key.get_secret_value() == "tvly-custom"

    @patch.dict(os.environ, {}, clear=True)
    def test_extra_fields_ignored(self):
        """.env 中无关字段应被忽略（extra='ignore'）"""
        with patch.dict(os.environ, {
            "LLM_API_KEY": "sk-test",
            "TAVILY_API_KEY": "tvly-test",
            "SOME_UNKNOWN_FIELD": "should-be-ignored",
        }):
            from config import APIConfig
            cfg = APIConfig()
            # 不应抛出 ValidationError
            assert cfg.llm_api_key.get_secret_value() == "sk-test"

    def test_temperature_bounds(self):
        """temperature 应在 0.0-2.0 范围内"""
        from config import APIConfig

        # 有效范围
        cfg = APIConfig(llm_temperature=0.0)
        assert cfg.llm_temperature == 0.0

        cfg2 = APIConfig(llm_temperature=2.0)
        assert cfg2.llm_temperature == 2.0

    @patch.dict(os.environ, {}, clear=True)
    def test_temperature_out_of_bounds_raises(self):
        """非法 temperature 值应抛出 ValidationError"""
        from pydantic import ValidationError
        from config import APIConfig

        with patch.dict(os.environ, {"LLM_TEMPERATURE": "3.0"}):
            with pytest.raises(ValidationError):
                APIConfig()


# ============================================================
# AgentConfig 测试
# ============================================================
class TestAgentConfig:
    """Agent 运行时配置测试"""

    def test_default_values(self):
        """默认值检查"""
        cfg = AgentConfig()

        # 搜索
        assert cfg.max_search_results == 10
        assert cfg.search_depth == "advanced"
        assert cfg.search_timeout == 15.0
        assert cfg.include_domains == []
        assert cfg.exclude_domains == []

        # 裁剪
        assert cfg.top_k_fragments == 5
        assert cfg.max_context_tokens == 3000
        assert cfg.dedup_window == 50

        # 会话
        assert cfg.max_history_turns == 10
        assert cfg.session_ttl_seconds == 3600

        # 重试
        assert cfg.max_retries == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 30.0

        # 限流
        assert cfg.max_requests_per_minute == 30
        assert cfg.max_searches_per_minute == 10

        # 调试
        assert cfg.verbose is False

    def test_mutable_fields(self):
        """配置字段应是独立可变的"""
        cfg = AgentConfig()

        cfg.max_search_results = 5
        assert cfg.max_search_results == 5

        cfg.verbose = True
        assert cfg.verbose is True

        cfg.include_domains = ["example.com"]
        assert cfg.include_domains == ["example.com"]

    def test_separate_instances_independent(self):
        """不同实例互不影响"""
        cfg1 = AgentConfig()
        cfg2 = AgentConfig()

        cfg1.verbose = True
        cfg1.max_search_results = 3

        assert cfg2.verbose is False
        assert cfg2.max_search_results == 10


# ============================================================
# 单例函数测试
# ============================================================
class TestSingletonFunctions:
    """get_api_config / get_agent_config 单例行为"""

    def test_get_api_config_returns_same_instance(self):
        """get_api_config() 应返回同一实例"""
        cfg1 = get_api_config()
        cfg2 = get_api_config()
        assert cfg1 is cfg2

    def test_get_agent_config_returns_same_instance(self):
        """get_agent_config() 应返回同一实例"""
        cfg1 = get_agent_config()
        cfg2 = get_agent_config()
        assert cfg1 is cfg2

    def test_api_and_agent_config_are_different_types(self):
        """两种配置是不同类型"""
        api_cfg = get_api_config()
        agent_cfg = get_agent_config()
        assert type(api_cfg) != type(agent_cfg)


# ============================================================
# 配置完整性测试
# ============================================================
class TestConfigCompleteness:
    """确保所有必需的配置项都存在"""

    def test_api_config_has_all_required_fields(self):
        """APIConfig 必需字段"""
        cfg = get_api_config()
        required_fields = [
            "llm_api_key", "llm_api_base", "llm_model", "llm_temperature",
            "tavily_api_key", "tavily_api_base",
        ]
        for field in required_fields:
            assert hasattr(cfg, field), f"缺少字段: {field}"

    def test_agent_config_has_all_required_fields(self):
        """AgentConfig 必需字段"""
        cfg = get_agent_config()
        required_fields = [
            "max_search_results", "search_depth", "search_timeout",
            "top_k_fragments", "max_context_tokens", "dedup_window",
            "max_history_turns", "session_ttl_seconds",
            "max_retries", "base_delay", "max_delay",
            "max_requests_per_minute", "max_searches_per_minute",
            "verbose",
        ]
        for field in required_fields:
            assert hasattr(cfg, field), f"缺少字段: {field}"

    def test_no_none_config_values(self):
        """所有配置值不应为 None（应有默认值）"""
        cfg = get_agent_config()
        for attr in dir(cfg):
            if not attr.startswith("_") and not callable(getattr(cfg, attr)):
                val = getattr(cfg, attr)
                # include_domains 和 exclude_domains 可以为空列表
                if attr in ("include_domains", "exclude_domains"):
                    assert isinstance(val, list), f"{attr} 应该是 list"
                else:
                    assert val is not None, f"{attr} 是 None"
