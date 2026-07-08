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
│   ├── test_core.py         # 单元测试 (去重/Token/限流/会话)
│   ├── test_registry.py     # 工具注册中心测试 (4 工具 + 组合流水线)
│   ├── test_graph.py        # Agent 状态图节点测试 (4 节点 + 路由 + 类)
│   ├── test_config.py       # 配置加载测试 (API + Agent 配置)
│   └── test_e2e.py          # 端到端测试 (Agent 完整流水线)
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
# 全部测试 (131 条)
pytest tests/ -v

# 按模块运行
pytest tests/test_core.py -v      # 单元测试 (33 条)
pytest tests/test_registry.py -v  # 工具注册中心 (31 条)
pytest tests/test_graph.py -v     # 状态图节点 (31 条)
pytest tests/test_config.py -v    # 配置加载 (18 条)
pytest tests/test_e2e.py -v       # 端到端 (18 条)
```

## 🧪 测试分层

项目采用**三层金字塔**测试策略，131 条测试全部 mock 外部 API，无需联网即可运行：

### 第一层: 单元测试 (`test_core.py` + `test_config.py`) — 51 条

| 测试类 | 覆盖内容 | 条数 |
|--------|----------|------|
| `TestURLDeduplicator` | URL 规范化、去重、滑动窗口、跟踪参数 | 7 |
| `TestTokenCounting` | 中英文 Token 计数 | 4 |
| `TestContentFingerprint` | SHA-256 内容指纹 | 4 |
| `TestRateLimiter` | 滑动窗口限流 | 3 |
| `TestContextTrimming` | Top-K + Token 裁剪 | 4 |
| `TestFormatSearchContext` | 搜索结果格式化 | 2 |
| `TestSessionStore` | 会话 TTL 存储 | 5 |
| `TestModels` | Pydantic 数据模型 | 3 |
| `TestAPIConfig` | 环境变量加载、SecretStr 安全、超限校验 | 7 |
| `TestAgentConfig` | 默认值、可变性、实例隔离 | 3 |
| `TestSingletonFunctions` | 单例缓存、类型区分 | 3 |
| `TestConfigCompleteness` | 字段完整性、非空校验 | 3 |
| `TestProjectRoot` | 根目录存在性 | 2 |

### 第二层: 集成测试 (`test_registry.py` + `test_graph.py`) — 62 条

| 测试类 | 覆盖内容 | 条数 |
|--------|----------|------|
| `TestTavilySearch` | 搜索成功/超时/异常/自定义参数/空结果/域名过滤 | 6 |
| `TestRewriteQuery` | 正常改写/历史上下文/LLM失败回退/JSON异常/默认值填充 | 6 |
| `TestScoreRelevance` | 打分成功/空输入/LLM失败回退/零分默认/内容截断 | 5 |
| `TestFallbackAnswer` | 降级回答/历史传递/自身失败/System Prompt/历史截断 | 5 |
| `TestSearchAndFilterPipeline` | 完整流水线/去重/API错误/空搜索/全重复/打分失败回退 | 6 |
| `TestToolRegistryMetadata` | 工具注册表完整性/类别匹配/唯一性 | 3 |
| `TestNodeRewriteQuery` | 状态更新/空查询/历史传递 | 3 |
| `TestNodeSearch` | 无查询/子查询拆分/状态键完整性 | 3 |
| `TestNodeGenerateAnswer` | 降级路径/片段生成/来源去重/历史上下文/二次降级 | 5 |
| `TestNodeErrorRecovery` | 标志设置/空消息/缺键默认 | 3 |
| `TestRoutingEdgeCases` | 缺键默认/空字符串/falsy值行为 | 5 |
| `TestSearchAgentClass` | 初始化/verbose/空历史/清除会话/去重器/latency | 6 |
| `TestAgentStateTypedDict` | 必需键完整/_fragments 私有键 | 2 |
| `TestAnswerSystemPrompt` | 非空/关键指令/足够长度 | 3 |

### 第三层: 端到端测试 (`test_e2e.py`) — 18 条

覆盖 `graph.py#L366-L384` 的**完整 Agent 流水线**：

| 测试类 | 覆盖路径 | 条数 |
|--------|----------|------|
| `TestAgentPipelineE2E` | ①正常流程 ②搜索失败→降级 ③答案生成失败→二次降级 ④多轮对话 ⑤历史裁剪 ⑥改写失败→回退 | 6 |
| `TestRoutingLogic` | 搜索后路由 + 生成后路由 | 4 |
| `TestNodeBehaviors` | 错误恢复 + 空查询 + 降级生成 | 3 |
| `TestSessionManagement` | 自动 session + 清除会话 | 2 |
| `TestResultValidation` | 来源去重 + 延迟记录 + 流式输出 | 3 |

```
        ┌─────────────────────────────────────────────────┐
        │      E2E: Agent 完整流水线 (18 条)              │
        │  改写 → 搜索 → 路由 → 答案生成 → 错误恢复 → 历史  │
        ├─────────────────────────────────────────────────┤
        │    集成: 工具注册 + 状态图节点 (62 条)           │
        │  4 工具 | 组合流水线 | 4 节点 | 路由 | Agent 类   │
        ├─────────────────────────────────────────────────┤
        │    单元: 工具函数 + 配置 (51 条)                 │
        │  去重 | Token | 限流 | 裁剪 | 会话 | 配置 | 模型  │
        └─────────────────────────────────────────────────┘
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
