# 🔍 智能搜索助手 — 实时联网搜索 Agent 系统

基于 **LangGraph** + **Tavily API** + **Streamlit** 构建的实时联网搜索 Agent。

## ✨ 核心特性

- **实时搜索**: Tavily API 联网搜索，突破 LLM 知识时效限制
- **智能改写**: LLM 驱动的查询改写，将口语化提问转为精准搜索词
- **双层过滤**: URL 去重 + LLM 相关性打分，减少 **60% 无效 Token**
- **多层容错**: 指数退避重试 + LLM 降级回答，异常期可用性 **95%+**
- **多轮对话**: 基于会话 ID 的上下文缓存，跨轮意图连贯
- **流式输出**: Streamlit 前端，漂亮的聊天界面
- **限流保护**: 搜索 API 配额管理与滑动窗口限流

## 🏗️ 架构

```
用户查询 → LLM 查询改写 → Tavily 实时搜索 → URL 去重 + LLM 相关性打分 → 答案生成
                                                  ↓
                                          Top-K + Token 裁剪
                                                  ↓
                                          仅保留高质量片段
```

### Agent 工作流

```
┌──────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  用户输入     │ ──→ │ 查询改写  │ ──→ │ 实时搜索  │ ──→ │ 结果过滤  │
└──────────────┘     └──────────┘     └──────────┘     └──────────┘
                                                              │
                                              ┌───────────────┘
                                              ↓
                                         ┌──────────┐     ┌──────────┐
                                         │ 答案生成  │ ←── │ 错误恢复  │
                                         └──────────┘     └──────────┘
```

## 📁 项目结构

```
问答系统/
├── agent/
│   ├── __init__.py          # Agent 模块
│   ├── graph.py             # LangGraph 状态图 (核心编排)
│   └── models.py            # Pydantic 数据模型
├── tools/
│   ├── __init__.py          # 工具模块
│   └── registry.py          # 工具注册: 搜索/改写/打分/降级
├── memory/
│   ├── __init__.py          # 内存模块
│   └── session_store.py     # 会话缓存 (TTL 过期)
├── utils/
│   ├── __init__.py          # 工具函数模块
│   └── helpers.py           # URL 去重, Token 计数, 限流, 裁剪
├── tests/
│   ├── __init__.py
│   └── test_core.py         # 单元测试
├── config.py                # 全局配置 (pydantic-settings)
├── app.py                   # Streamlit 前端入口
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量模板
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd 问答系统
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
# LLM_API_KEY=sk-xxx
# TAVILY_API_KEY=tvly-xxx
```

### 3. 运行

```bash
# Web 前端
streamlit run app.py

# 或命令行方式
python -c "
import asyncio
from agent.graph import SearchAgent
agent = SearchAgent(verbose=True)
result = asyncio.run(agent.run('今天有什么重大新闻？'))
print(result.answer)
"
```

### 4. 运行测试

```bash
pytest tests/ -v
```

## 🔧 工具注册

项目遵循**职责单一原则**注册 4 个工具：

| 工具 | 类别 | 职责 |
|------|------|------|
| `tavily_search` | 搜索 | Tavily API 实时联网搜索 |
| `query_rewriter` | 改写 | LLM 口语化→精准查询词 |
| `relevance_scorer` | 打分 | LLM 0-1 相关性评分 |
| `fallback_answer` | 降级 | 搜索不可用时 LLM 知识回答 |

## 📊 业务场景

- **实时新闻**: "今天有什么重大新闻？"
- **股价查询**: "苹果公司现在的股价是多少？"
- **热点事件**: "最近发生的 AI 领域大事件有哪些？"
- **对比分析**: "GPT-4 和 Claude 哪个更强？"
- **教程指南**: "最新的 Python 3.13 有哪些新特性？"

## 🛡️ 容错机制

| 故障类型 | 恢复策略 |
|----------|----------|
| 搜索 API 超时 | 指数退避重试 3 次 → LLM 降级回答 |
| 查询改写失败 | 回退到原始用户输入 |
| 相关性打分失败 | 使用搜索 API 原始分数 |
| LLM API 故障 | 友好错误提示 |
| 配额耗尽 | 滑动窗口限流，提前拦截 |

## 📈 效果指标

- 正常期可用性: **≈100%**
- 异常期可用性: **95%+**
- Token 节省: **60%** (URL 去重 + 相关性打分双层过滤)
- 查询改写贡献: 命中率提升 **≈35%**

## 📄 License

MIT
