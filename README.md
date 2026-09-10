# RAG 知识库问答系统

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![Chroma](https://img.shields.io/badge/Vector_DB-ChromaDB-4B32C3)
![Tests](https://img.shields.io/badge/tests-152_passed-brightgreen)

一个基于 FastAPI 的中文知识库问答系统，支持文档上传、自动切分、向量化入库、混合检索、重排序、LLM 生成回答，并配套完整的离线评测体系。

## 核心能力

- 检索链路：BGE Embedding + BM25 混合召回、RRF 融合、BGE Reranker 精排与低分拒答。
- 对话能力：多轮会话、查询改写、SSE 流式输出、规则/LLM Router。
- Agent 能力：标准 Function Calling Tool Agent、工具白名单、参数校验、并行调用与有限步数控制。
- 协议与记忆：MCP Server/Client 标准化工具调用；跨会话长期记忆的抽取、召回和 Prompt 注入。
- 工程能力：SQLite/Chroma 双写补偿、Redis 会话、健康检查、Docker、超时重试、限流、熔断、监控指标与 CI。
- 效果评测：人工标注评测集，计算 Recall@K、Precision@K、MRR 与负样本拒答率。

## 快速开始

完整的环境准备、依赖安装和评测命令见下文“如何运行”。

```powershell
# 启动 FastAPI 服务
python -m uvicorn main:app --reload

# 运行全部测试
python -m pytest -q tests
```

项目默认使用本地 BGE Embedding 和 BGE Reranker，模型文件不包含在仓库中；LLM 调用通过 `OPENAI_API_KEY` 环境变量配置。

## 项目定位

这是一个用于学习 RAG 全链路和工程实践的个人项目。目标不是做一个通用问答产品，而是完整走通：

```text
文档入库 → 检索召回 → 重排序 → 生成回答 → 效果评测
```

并沉淀可复现、可对比的评测指标。

## 技术栈

| 层次 | 技术 |
| --- | --- |
| Web 框架 | FastAPI、Pydantic v2、Uvicorn |
| 关系数据库 | SQLite、SQLAlchemy |
| 向量数据库 | ChromaDB |
| Embedding | bge-small-zh-v1.5（512 维） |
| 关键词检索 | BM25（rank-bm25）+ jieba 分词 |
| 重排序 | bge-reranker-v2-m3（CrossEncoder） |
| LLM | DeepSeek（OpenAI 兼容接口，支持 Function Calling） |
| 测试 | pytest、FastAPI TestClient、依赖覆盖 |

## 系统架构

```text
┌────────────┐     ┌──────────────┐     ┌─────────────┐
│  FastAPI   │────▶│   routers    │────▶│  services   │
│   main.py  │     │ documents.py │     │ vector_store│
│  lifespan  │     │  chat.py     │     │  reranker   │
└────────────┘     └──────────────┘     │  llm_service │
                                        └─────────────┘
                                             │      │
                              ┌──────────────┘      └──────────────┐
                              ▼                                     ▼
                     ┌──────────────┐                      ┌──────────────┐
                     │   ChromaDB   │                      │  DeepSeek    │
                     │  向量 + 元数据│                      │  LLM API     │
                     └──────────────┘                      └──────────────┘

SQLite：记录文档状态（processing / indexed / failed）
Chroma：保存 chunk 向量、文本和来源元数据
```

## 核心检索流程

### 文档入库

```text
上传文件
  → 文本切分（chunk_size / chunk_overlap）
  → 生成 chunk_id 和 metadata
  → Embedding 向量化
  → 写入 ChromaDB
  → SQLite 记录文档状态
```

### 问答检索

```text
用户提问
  → Agent Router：判断是否需要检索
  │     ├── direct_answer：闲聊/翻译类，直接生成
  │     ├── refuse：明确拒答
  │     └── retrieve：
  │           → 查询改写（LLM）
  │           → 向量检索 + BM25 关键词检索
  │           → RRF 融合，先召回 top_k × 3
  │           → Reranker 重排序，选最终 top_k
  │           → LLM 根据上下文生成回答
  → 返回答案 + 来源引用
```

## 目录结构

```text
main.py                     FastAPI 入口、lifespan
config.py                   Pydantic Settings 配置
database.py                 SQLite 引擎与 get_db
models.py                   文档状态表
schemas.py                  请求/响应模型
logging_config.py           日志配置

routers/documents.py        上传、列表、删除接口
routers/chat.py             问答接口与依赖注入

services/vector_store.py    向量库、BM25、混合检索
services/reranker.py        重排序模型
services/llm_service.py     LLM 调用与查询改写
services/agent_state.py     Agent 状态与结构化路由决策
services/agent_router.py    rules / LLM 两种 Router
services/tool_agent.py      标准 Function Calling Tool Agent
services/mcp_client.py      MCP stdio 客户端
services/mcp_tools.py       MCP 工具 → Function Calling 工具适配
services/memory_store.py    长期记忆存储（内存/Redis）
services/memory_extractor.py 规则版记忆抽取
services/memory_service.py  记忆召回与 Prompt 格式化
services/dependencies.py    共享单例
services/text_splitter.py   文本切分
services/metrics.py         检索评测指标

mcp_server/kb_server.py     最小 MCP Server（JSON-RPC + stdio）

evaluate_retrieval.py       检索层离线评测
evaluate_agent_router.py    Agent Router（rules vs LLM）对照实验
evaluate_generation.py      端到端生成层评测
evaluation_dataset.json     人工标注评测集
analyze_rag_logs.py         RAG/Agent 日志指标聚合

tests/                      单元测试与接口测试
```

## 评测体系

### 评测集

```text
20 道正样本：知识库中能找到答案的问题
4 道负样本：知识库中无法回答的问题
```

每道正样本都人工标注了 `relevant_chunk_ids`，作为检索层的标准答案。

### 评测指标

检索层：

```text
Recall@K、Precision@K、MRR、拒答率
```

生成层：

```text
Faithfulness、Context Recall、Answer Relevance
```

生成层使用 LLM-as-judge，支持结构化输出和多次采样取多数，降低随机波动。

### 真实评测结果

检索层（20 正样本 + 4 负样本，top_k=3，Reranker threshold=0.5）：

```text
strategy              Recall@3   Precision@3   MRR   拒答率
hybrid                 1.000        0.517     0.942   0.000
hybrid + reranker      1.000        0.842     1.000   1.000
```

结论：Reranker 不改变召回，但把 Precision 从 51.7% 提升到 84.2%，MRR 从 0.942 提升到 1.0，并在 4 道负样本上全部正确拒答。

Agent Router（rules vs LLM，20 正样本 + 4 负样本，top_k=3，threshold=0.5）：

```text
mode   Recall@3   Precision@3   MRR   负样本空返回率  平均路由耗时  平均总耗时
rules   1.000       0.842      1.000      1.000         0.0ms      1125.4ms
llm     0.950       0.792      0.950      1.000       1183.3ms     2182.6ms
```

结论：LLM Router 在本评测集中把 “CNN 的全称是什么” 误判为“常识性事实，无需检索”，绕过知识库直接回答，导致 Recall 从 1.0 降到 0.95。这正说明 Router 必须配合规则兜底和检索层校验，不能无条件信任模型决策。

生成层（5 道正样本，3 次采样）：

```text
Faithfulness = 0.800
Context Recall = 0.800
Answer Relevance = 1.000
```

注意：当前知识库只有 77 个 chunk，题目与知识库内容高度相关，因此 Recall 偏高。这些数字用于验证评测链路，不代表真实业务表现。

## Function Calling Tool Agent

除了 `/chat/ask` 的固定 RAG 流程，项目还提供一个标准 Function Calling 演示接口：

```text
POST /chat/agent/tool
```

模型通过 OpenAI/DeepSeek 兼容的 `tools` 协议声明 `search_knowledge_base` 工具；
需要知识库事实时返回 `tool_calls`，代码执行检索并把结果按 `tool_call_id` 回填，
模型再继续推理直到给出最终答案。

工程边界：

- 工具名来自白名单，未知工具直接返回错误给模型；
- 工具参数必须是合法 JSON 对象，解析失败不崩溃；
- 循环受 `max_steps=3` 限制，防止模型反复调用工具；
- 真实验证中模型一次返回 2 个并行检索调用，拿到上下文后正确回答“CNN 的全称是什么”。

相关测试：`tests/test_tool_agent.py`（循环与边界）、
`tests/test_tool_agent_api.py`（接口层）。

## MCP Server

项目把知识库检索暴露为标准 MCP 工具，协议层使用 JSON-RPC 2.0 + stdio，
不依赖官方 SDK 也能直接运行：

```powershell
python -m mcp_server.kb_server
```

支持 `initialize`、`ping`、`tools/list`、`tools/call`，内置工具：

```text
search_knowledge_base       检索知识库片段
list_knowledge_documents    列出已入库文档
```

客户端调用示例：

```python
from services.mcp_client import StdioMCPClient

with StdioMCPClient() as client:
    client.initialize()
    tools = client.list_tools()
    contexts = client.call_tool(
        "search_knowledge_base",
        {"query": "CNN 的全称是什么", "top_k": 3},
    )
```

Function Calling 解决“模型怎么请求工具”，MCP 解决“工具怎么被标准化复用”。

### MCP 接入 Tool Agent

`MCP_ENABLED=true` 时，`/chat/agent/tool` 的工具声明和执行都走 MCP Server：

```text
MCP tools/list ──转换──▶ Function Calling tools ──▶ LLM 返回 tool_calls
                                                          │
                                    MCP tools/call ◀──────┘
```

转换只改字段名，不改 schema 内容：

```text
MCP                    Function Calling
name            →      function.name
description     →      function.description
inputSchema     →      function.parameters
```

这样工具只需在 MCP Server 里声明一次，Agent 不必手写工具定义，工具白名单
也直接取自 `tools/list`，声明和权限不会跑偏。默认 `MCP_ENABLED=false`，保持
单进程开发；开启后 MCP Server 以子进程懒加载，工具列表缓存一次，应用关闭
时随 lifespan 一起释放。MCP Server 启动失败或返回空工具列表时，接口自动
回退到内置本地检索工具。

## 长期记忆

`/chat/ask` 支持可选的 `user_id`。系统会在检索前召回该用户的长期记忆并注入
Prompt，在生成成功后从用户问题中抽取新的记忆：

```text
用户问题
  → 记忆召回
  → 检索 + Reranker
  → 生成（注入 user_memory）
  → 记忆抽取与写回
```

记忆存储支持内存和 Redis 两种后端；生产环境建议使用 Redis 以支持多实例共享。

## 技术难点与解决方案

### 1. Anaconda 旧 UCRT 导致 Torch DLL 加载失败

现象：`import torch` 报 `WinError 1114`，`c10.dll` 初始化失败。

原因：虚拟环境基于 Anaconda 创建，Anaconda 自带一套 Windows 10 初版的旧 UCRT 运行时（`ucrtbase.dll 10.0.10240`），与 Torch 2.x 冲突。

解决：改用独立的官方 Python 3.11 创建新虚拟环境，使用系统新版本 UCRT，问题消失。

### 2. SQLite 与 ChromaDB 双写一致性

问题：文档状态在 SQLite，向量在 Chroma，两者无法用同一个事务保证原子性。

方案：

```text
先写入新版本
新版本成功后才删除旧版本
失败时清理新版本残留，并记录 failed 状态
```

### 3. async 接口中调用同步耗时函数

问题：在 `async def` 中直接调用 Embedding、Reranker、LLM 会阻塞事件循环。

方案：使用 `fastapi.concurrency.run_in_threadpool` 把耗时操作放入工作线程。

### 4. Reranker 阈值与低分拒答

问题：没有阈值时，知识库中不相关的内容也会被返回，导致无法拒答。

方案：为 Reranker 增加分数阈值，负样本被过滤为空，实现“没有可靠依据就拒答”。

### 5. LLM-as-judge 结果不稳定

问题：同一个评测跑两次，指标波动较大。

方案：让 LLM 输出结构化 JSON，并对同一判断多次采样取多数，降低随机性。

## 如何运行

安装依赖：

```powershell
.\.venv311\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

启动服务：

```powershell
.\.venv311\Scripts\python.exe -m uvicorn main:app --reload
```

运行测试：

```powershell
.\.venv311\Scripts\python.exe -m pytest -q tests
```

健康检查：

```text
GET /health/live   # 进程存活检查
GET /health/ready  # 模型与向量库就绪检查
```

监控指标：

```text
GET /metrics       # Prometheus 兼容文本格式
```

当前自动统计 HTTP 请求总数、状态码分布、请求耗时和 P95 延迟。
压测时重点观察 2xx/4xx/5xx 比例、平均延迟和 P95，而不是只看单次请求速度。

## CI 自动化

项目通过 `.github/workflows/ci.yml` 自动执行：

```text
安装依赖 → 编译关键模块 → 运行 100% Fake/单元测试 → 上传 JUnit 报告
```

CI 不调用真实 LLM、不需要 API Key，也不加载本地模型；真实模型评测通过本地
`evaluate_*.py` 脚本单独执行，避免把密钥和知识库内容发送到 CI 环境。

## Docker 部署

项目提供了 [Dockerfile](D:/software/Pycharm/Project/test1/fastApiProject/Dockerfile)
和 [docker-compose.yml](D:/software/Pycharm/Project/test1/fastApiProject/docker-compose.yml)。
镜像只包含代码和 Python 依赖，Embedding/Reranker 模型通过只读挂载提供，
SQLite、Chroma、上传文件和日志通过 volume 持久化。

准备 `.env` 后运行：

```powershell
docker compose up --build
```

查看容器状态：

```powershell
docker compose ps
```

其中 `/health/ready` 会作为容器健康检查；模型尚未加载完成时，容器仍然运行，
但健康状态不会变为 healthy。不要把 `.env`、模型文件和本地数据库复制进镜像。

## 缓存设计

当前为查询改写接入了可配置的进程内 TTL 缓存。缓存键包含原始问题、会话历史、
模型名称和 `KNOWLEDGE_BASE_VERSION`，因此知识库版本变化后不会误用旧结果。
进程内缓存适合单进程演示；多 worker 或多实例部署时应替换为 Redis。

## 稳定性保护

问答接口支持可配置的固定窗口限流；超过限制返回 `429` 和 `Retry-After`。
LLM 调用具有总次数上限、指数退避和超时边界；Tool Agent 具有总耗时上限。
项目还提供可复用的熔断器，连续外部依赖失败后暂时拒绝调用，恢复窗口后尝试半开探测。

## 会话持久化

默认 `CONVERSATION_STORE_BACKEND=memory`，适合测试和单进程开发；Docker Compose
会启动 Redis，并切换为 `CONVERSATION_STORE_BACKEND=redis`，使不同 worker 共享会话历史。
会话使用 TTL 自动过期；Redis 连接失败会返回明确的存储错误，不会伪装成空历史。

运行检索评测：

```powershell
.\.venv311\Scripts\python.exe evaluate_retrieval.py --dataset evaluation_dataset.json --top-k 3 --threshold 0.5
```

运行 Agent Router 对照实验（llm 模式需要真实 API）：

```powershell
.\.venv311\Scripts\python.exe evaluate_agent_router.py --top-k 3 --threshold 0.5
```

运行固定 RAG 与 Function Calling Tool Agent 的系统级对照实验（需要启动 FastAPI，并使用真实 API）：

```powershell
.\.venv311\Scripts\python.exe evaluate_fixed_vs_tool_agent.py --top-k 3
```

报告会记录同一批评测题上的 Recall/Precision/MRR、负样本空上下文率、平均耗时、P95 延迟，以及 Tool Agent 的平均工具调用次数和步骤数。

运行生成评测：

```powershell
.\.venv311\Scripts\python.exe evaluate_generation.py --dataset evaluation_dataset.json --top-k 3 --threshold 0.5 --judge-samples 3
```

## 简历可写的项目亮点

1. 独立实现 RAG 全链路：文档切分、向量化、混合检索、重排序、LLM 生成。
2. 混合检索使用“向量检索 + BM25 + RRF 融合”，兼顾语义和关键词。
3. 引入 Reranker 阈值机制，实现低分拒答，负样本拒答率达到 100%。
4. 建立分层离线评测：检索层 Recall/Precision/MRR/拒答率，生成层 Faithfulness/Context Recall/Answer Relevance。
5. 使用依赖注入和 Fake 替身实现不依赖真实模型与外部 API 的接口测试。
6. 解决 Windows 下 Anaconda 与 Torch 的 DLL 冲突，具备真实环境排障经验。
7. 引入 Agentic RAG 状态机：Router 决策、有界检索循环、AgentState 状态记录、`max_attempts` 防止死循环。
8. Router 支持规则与 LLM 两种模式：LLM 输出结构化 JSON 决策，解析失败自动回退规则；用 24 道评测题做了 rules vs llm 对照实验。
9. 普通与流式接口共用同一套 Agent 状态流，并通过日志记录 route_action、检索次数、阶段耗时，支持按 Router 模式聚合分析。
10. 实现标准 Function Calling Tool Agent：LLM 通过 tools 协议声明检索工具，支持多工具调用、非法参数兜底、最大步数限制，真实调用验证通过。
