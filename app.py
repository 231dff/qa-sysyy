"""
智能搜索助手 — Streamlit 前端
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from agent.graph import SearchAgent
from agent.models import GeneratedAnswer

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="智能搜索助手",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 初始化 Session State
# ============================================================
def init_session():
    """初始化 Streamlit 会话状态"""
    if "agent" not in st.session_state:
        st.session_state.agent = SearchAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0


init_session()


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🔍 智能搜索助手")
    st.markdown("---")

    st.markdown("### ⚙️ 设置")

    # 模型选择
    model = st.selectbox(
        "LLM 模型",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "claude-3-haiku"],
        index=0,
        help="选择用于查询改写和答案生成的模型",
    )

    # 搜索设置
    search_depth = st.radio(
        "搜索深度",
        options=["basic", "advanced"],
        index=1,
        help="basic: 快速搜索; advanced: 深度搜索（更多结果但更慢）",
    )

    top_k = st.slider(
        "Top-K 片段数",
        min_value=2,
        max_value=10,
        value=5,
        help="保留最相关的 K 个搜索结果片段",
    )

    st.markdown("---")

    # 统计
    st.markdown("### 📊 统计")
    st.metric("查询次数", st.session_state.total_queries)
    st.metric("Token 消耗", f"{st.session_state.total_tokens:,}")

    st.markdown("---")

    # 操作按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 新会话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent.clear_session(st.session_state.session_id)
            st.session_state.session_id = f"session_{datetime.now().strftime('%H%M%S')}"
            st.rerun()
    with col2:
        if st.button("🗑️ 清历史", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent.clear_session(st.session_state.session_id)
            st.rerun()

    st.markdown("---")
    st.caption("Powered by LangGraph + Tavily API + Streamlit")
    st.caption(f"会话 ID: `{st.session_state.session_id}`")


# ============================================================
# 主聊天界面
# ============================================================
st.title("🔍 智能搜索助手")
st.caption("实时联网搜索 · 可信答案生成 · 多轮对话理解")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 参考来源"):
                for src in msg["sources"]:
                    st.markdown(f"- [{src['title'] or src['url']}]({src['url']})")
        if msg.get("meta"):
            st.caption(msg["meta"])


# ============================================================
# 用户输入
# ============================================================
if prompt := st.chat_input("输入你的问题，例如：今天有什么重大新闻？"):

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 执行搜索
    with st.chat_message("assistant"):
        # 进度占位
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()

        status_placeholder.markdown("🔄 正在**改写查询**...")

        # 异步执行 Agent
        try:
            with st.spinner(""):
                result: GeneratedAnswer = asyncio.run(
                    st.session_state.agent.run(
                        user_query=prompt,
                        session_id=st.session_state.session_id,
                    )
                )
            status_placeholder.empty()
        except Exception as exc:
            status_placeholder.error(f"❌ 执行失败: {str(exc)}")
            result = None

        # 渲染答案
        if result:
            answer_placeholder.markdown(result.answer)

            # 显示来源
            if result.sources:
                with sources_placeholder.expander("📚 参考来源", expanded=False):
                    for i, src in enumerate(result.sources, 1):
                        st.markdown(f"**[{i}]** [{src.url}]({src.url})")
                        if src.snippet:
                            st.caption(src.snippet[:300])

            # 元信息
            status_icon = "⚠️" if result.is_fallback else "✅"
            confidence_pct = f"{result.confidence:.0%}"
            st.caption(
                f"{status_icon} 置信度: {confidence_pct} | "
                f"耗时: {result.latency_ms:.0f}ms | "
                f"Token: {result.tokens_used} | "
                f"降级回答: {'是' if result.is_fallback else '否'}"
            )

            # 更新统计
            st.session_state.total_queries += 1
            st.session_state.total_tokens += result.tokens_used

            # 保存消息
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "sources": [
                    {"url": s.url, "title": s.title}
                    for s in result.sources
                ],
                "meta": (
                    f"{status_icon} 置信度: {confidence_pct} | "
                    f"耗时: {result.latency_ms:.0f}ms | "
                    f"Token: {result.tokens_used}"
                ),
            })

            # 强制限制消息条数
            max_messages = 20
            if len(st.session_state.messages) > max_messages:
                st.session_state.messages = st.session_state.messages[-max_messages:]

            st.rerun()
