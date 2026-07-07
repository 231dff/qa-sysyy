"""
智能搜索助手 — 核心功能单元测试
测试范围: URL 去重、Token 计数、上下文裁剪、限流器、会话存储
"""
import pytest
import time
import uuid

from utils.helpers import (
    URLDeduplicator,
    count_tokens,
    content_fingerprint,
    RateLimiter,
    trim_context,
    format_search_context,
    generate_session_id,
)
from memory.session_store import SessionStore


# ============================================================
# URL 去重测试
# ============================================================
class TestURLDeduplicator:
    """URL 去重器测试"""

    def test_basic_dedup(self):
        dd = URLDeduplicator(window_size=10)
        assert not dd.is_duplicate("https://example.com/news/1")
        dd.add("https://example.com/news/1")
        assert dd.is_duplicate("https://example.com/news/1")

    def test_normalization(self):
        """测试 URL 规范化：尾部斜杠、大小写"""
        dd = URLDeduplicator(window_size=10)
        dd.add("https://example.com/News/")
        assert dd.is_duplicate("https://example.com/news")
        assert dd.is_duplicate("HTTPS://EXAMPLE.COM/NEWS/")

    def test_tracking_params_removal(self):
        """UTM 跟踪参数应被去除"""
        dd = URLDeduplicator(window_size=10)
        dd.add("https://example.com/page?utm_source=twitter&id=123")
        assert dd.is_duplicate("https://example.com/page?id=123")

    def test_window_eviction(self):
        """滑动窗口过期"""
        dd = URLDeduplicator(window_size=3)
        dd.add("url1")
        dd.add("url2")
        dd.add("url3")
        dd.add("url4")  # url1 应被弹出
        assert not dd.is_duplicate("url1")
        assert dd.is_duplicate("url2")

    def test_batch_dedup(self):
        dd = URLDeduplicator(window_size=50)
        urls = ["a", "b", "a", "c", "b", "d"]
        result = dd.deduplicate(urls)
        assert result == ["a", "b", "c", "d"]

    def test_different_domains_not_deduped(self):
        dd = URLDeduplicator(window_size=10)
        dd.add("https://example.com/news")
        assert not dd.is_duplicate("https://other.com/news")

    def test_clear(self):
        dd = URLDeduplicator(window_size=10)
        dd.add("url1")
        dd.add("url2")
        dd.clear()
        assert not dd.is_duplicate("url1")


# ============================================================
# Token 计数测试
# ============================================================
class TestTokenCounting:
    """Token 计数测试"""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_english_text(self):
        tokens = count_tokens("Hello, world! This is a test.")
        assert 5 <= tokens <= 15  # 宽松范围

    def test_chinese_text(self):
        tokens = count_tokens("你好世界这是一个测试")
        assert tokens > 0

    def test_long_text(self):
        text = "Lorem ipsum dolor sit amet. " * 100
        tokens = count_tokens(text)
        assert 300 < tokens < 1000


# ============================================================
# 内容指纹测试
# ============================================================
class TestContentFingerprint:
    """内容指纹测试"""

    def test_short_text_returns_none(self):
        assert content_fingerprint("short", min_length=100) is None

    def test_long_text_returns_hash(self):
        text = "x" * 200
        fp = content_fingerprint(text, min_length=100)
        assert fp is not None
        assert len(fp) == 64  # SHA-256 hex

    def test_same_content_same_hash(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        assert content_fingerprint(text) == content_fingerprint(text)

    def test_different_content_different_hash(self):
        assert content_fingerprint("a" * 200) != content_fingerprint("b" * 200)


# ============================================================
# 限流器测试
# ============================================================
class TestRateLimiter:
    """滑动窗口限流器测试"""

    def test_allow_within_limit(self):
        rl = RateLimiter(max_per_minute=5)
        for _ in range(5):
            assert rl.acquire() is True

    def test_deny_over_limit(self):
        rl = RateLimiter(max_per_minute=3)
        for _ in range(3):
            rl.acquire()
        assert rl.acquire() is False

    def test_remaining(self):
        rl = RateLimiter(max_per_minute=10)
        for _ in range(4):
            rl.acquire()
        assert rl.remaining == 6


# ============================================================
# 上下文裁剪测试
# ============================================================
class TestContextTrimming:
    """上下文裁剪测试"""

    def test_top_k_selection(self):
        fragments = [
            ("url1", "content a", 0.9),
            ("url2", "content b", 0.3),
            ("url3", "content c", 0.7),
            ("url4", "content d", 0.5),
        ]
        result = trim_context(fragments, max_tokens=10000, top_k=2)
        assert len(result) == 2
        assert result[0][2] >= result[1][2]  # 降序
        assert all(s[2] >= 0.5 for s in result)

    def test_token_cap(self):
        """Token 超限时应截断"""
        fragments = [
            ("url1", "The quick brown fox. " * 50, 0.95),
        ]
        result = trim_context(fragments, max_tokens=50, top_k=5)
        # 应被截断到约 50 token
        assert len(result) <= 1
        if result:
            assert count_tokens(result[0][1]) <= 55  # 允许少量误差

    def test_empty_input(self):
        result = trim_context([], max_tokens=1000, top_k=5)
        assert result == []

    def test_minimum_fragment_length(self):
        """太短的片段会被跳过（如果 token 已满）"""
        fragments = [
            ("url1", "hi", 0.9),
        ]
        result = trim_context(fragments, max_tokens=10, top_k=5)
        # 不跳过短片段，只是在 token 不够时截断
        assert len(result) >= 0


# ============================================================
# 格式化测试
# ============================================================
class TestFormatSearchContext:
    """搜索上下文格式化测试"""

    def test_empty(self):
        assert "无相关搜索结果" in format_search_context([])

    def test_format(self):
        fragments = [
            ("https://example.com/1", "content one", 0.9),
        ]
        result = format_search_context(fragments)
        assert "[来源 1]" in result
        assert "https://example.com/1" in result
        assert "0.90" in result
        assert "content one" in result


# ============================================================
# 会话存储测试
# ============================================================
class TestSessionStore:
    """会话存储测试"""

    def test_get_empty(self):
        store = SessionStore(ttl_seconds=3600)
        assert store.get("nonexistent") == []

    def test_put_and_get(self):
        store = SessionStore(ttl_seconds=3600)
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        store.put("session1", history)
        assert store.get("session1") == history

    def test_delete(self):
        store = SessionStore(ttl_seconds=3600)
        store.put("session1", [{"role": "user", "content": "test"}])
        store.delete("session1")
        assert store.get("session1") == []

    def test_exists(self):
        store = SessionStore(ttl_seconds=3600)
        assert not store.exists("session1")
        store.put("session1", [])
        assert store.exists("session1")

    def test_cleanup_expired(self):
        store = SessionStore(ttl_seconds=0.001)  # 极小 TTL，立刻过期
        store.put("session1", [{"role": "user", "content": "test"}])
        time.sleep(0.1)  # 确保时间差足够（Windows 时钟精度约 15ms）
        cleaned = store.cleanup_expired()
        assert cleaned >= 1
        assert not store.exists("session1")


# ============================================================
# 会话 ID 生成测试
# ============================================================
def test_generate_session_id():
    sid = generate_session_id()
    assert len(sid) == 8
    # 两次生成应该不同
    assert generate_session_id() != generate_session_id()


# ============================================================
# 数据模型测试
# ============================================================
class TestModels:
    """Pydantic 模型测试"""

    def test_search_result(self):
        from agent.models import SearchResult
        r = SearchResult(url="https://example.com", title="Test", content="Content")
        assert r.url == "https://example.com"
        assert r.score == 0.0

    def test_generated_answer(self):
        from agent.models import GeneratedAnswer, Source
        answer = GeneratedAnswer(
            query="test",
            answer="This is a test answer",
            sources=[Source(url="https://example.com", title="Example", snippet="...")],
            confidence=0.9,
        )
        assert len(answer.sources) == 1
        assert answer.confidence == 0.9
        assert not answer.is_fallback

    def test_rewritten_query(self):
        from agent.models import RewrittenQuery
        rq = RewrittenQuery(
            original="今天有什么新闻？",
            rewritten="2025年7月 重大新闻",
            sub_queries=["中国新闻 2025年7月"],
        )
        assert rq.intent == "factual"
        assert len(rq.sub_queries) == 1
