# RAG 知识库问答系统：项目串讲

## 1. 项目一句话介绍

这是一个基于 FastAPI 的中文 RAG 知识库问答系统，完整实现了文档入库、向量化、混合检索、Reranker 重排、低分拒答、LLM 生成、多轮会话、Agent、Function Calling、离线评测和服务化部署。

项目的核心目标不是“调用一次大模型”，而是把一次问答拆成可观测、可测试、可评估的完整链路。

## 2. 总体架构

```text
用户请求
   │
   ▼
FastAPI 路由
   │
   ├── 参数校验（Pydantic）
   ├── Router：retrieve / direct_answer / refuse
   ├── 查询改写与历史会话
   ├── Embedding 向量检索
   ├── BM25 关键词检索
   ├── RRF 融合
   ├── Reranker 重排与阈值过滤
   ├── LLM 生成 / Function Calling
   ├── 输出安全检查
   └── 保存会话、日志、指标
        │
        ├── SQLite：文档状态和业务元数据
        ├── Chroma：chunk、向量、metadata
        ├── Redis：多实例共享会话（可选）
        └── DeepSeek：查询改写和答案生成
```

## 3. 文档入库流程

```text
上传文件
  → 生成 document_id
  → 切分为多个 chunk
  → 给每个 chunk 保存 source / document_id / chunk_id
  → 生成 Embedding
  → 写入 Chroma
  → 更新 SQLite 状态为 indexed
```

SQLite 和 Chroma 是两个独立系统，不能共享同一个事务。因此更新文档时采用补偿思路：先写新版本，确认新版本成功后再清理旧版本；失败时删除新版本残留，并把 SQLite 状态记录为 failed。

## 4. 检索流程

```text
用户问题
  → 查询改写
  → 向量检索 top_k × 3
  → BM25 检索 top_k × 3
  → RRF 按排名融合
  → Reranker 重新计算 query-document 相关性
  → 分数阈值过滤
  → 保留最终 top_k
```

各组件职责不同：

- Embedding 检索：理解语义相似性；
- BM25：补充关键词、缩写和精确术语匹配；
- RRF：融合不同检索器的排名，避免直接相加不同量纲的分数；
- Reranker：对较小候选集做更精细的 query-document 交互判断；
- 阈值：没有可靠依据时拒绝交给 LLM，降低幻觉风险。

## 5. 普通 RAG 请求流程

`POST /chat/ask` 的主要状态为：

```text
ROUTE → RETRIEVE → CHECK → GENERATE → VALIDATE → SAVE
```

关键行为：

1. Router 不确定时默认检索；
2. 查询改写失败时回退原问题；
3. 改写问题没有结果时，最多用原问题补检索一次；
4. 没有可靠 context 时不调用 LLM；
5. LLM 失败或输出不安全时不保存 assistant 空消息；
6. 只有生成成功并通过检查后才保存一轮会话。

## 6. Agent 与 Function Calling

普通 Agent Router 使用结构化决策：

```text
LLM 输出 action
  → Pydantic 校验
  → 代码 if/else 执行固定分支
```

Function Calling 使用标准协议：

```text
tools
  → assistant.tool_calls
  → 后端校验白名单和 JSON 参数
  → 执行 search_knowledge_base
  → role=tool 回填结果
  → LLM 再次决策
```

代码始终掌握最终权限：工具白名单、参数校验、最大步数、超时和异常处理都不能交给模型自觉遵守。

工具来源可以切换成 MCP：`MCP_ENABLED=true` 时，工具声明来自 MCP `tools/list`
（`inputSchema` 改名为 `parameters`），执行转发到 MCP `tools/call`，白名单直接
由声明推导。对模型来说协议没变，变的是工具由谁声明、由谁执行。

追问「MCP 和 Function Calling 冲突吗」：不冲突，一个管工具怎么被描述和复用，
一个管模型怎么发起调用，两者串联成一条链路。

追问「MCP 子进程挂了怎么办」：工具列表拿不到就回退内置本地工具；单次调用
异常会转成 `role=tool` 的错误内容交回模型，让模型自己决定改问法还是放弃，
而不是把异常直接变成 500。

## 7. Planner / Verifier

检索层从"固定脚本"升级成有界状态机：

```text
ROUTE → RETRIEVE → VERIFY → REPLAN → RETRIEVE → ... → GENERATE → VALIDATE
```

Verifier 回答"够不够"，Planner 回答"下一步干什么"。判定分四档：
`empty`（没候选）/ `low_score`（全被阈值拦）/ `low_coverage`（有文档但跑题）
/ `sufficient`（放行）。补救策略按失败类型选：空结果就回原问题，完全跑题
才换关键词查询，部分覆盖直接降级生成。

追问「为什么要分工，不能一个函数搞定」：分工让"判断"和"决策"可以独立替换和
独立评测。判断层换成 LLM 验证器、决策层换成别的策略，互不影响；实测校准也
只针对判断层，指标干净。

追问「怎么防死循环」：三重保险——次数上限在 AgentState、Planner 只能返回白名单
动作、主循环里还有一层"超过预算就强制收尾"的兜底。测试里专门注入了一个只会
说"继续检索"的坏 Planner，验证它照样只能搜两次。

追问「这个模块到底带来了什么」：做了校准实验。阈值 0.5 时它没有额外收益（阈值
已经够用），所以默认配置下应该关掉；阈值关闭时它把"知识库没答案却照常送给模型"
的比例从 4/4 降到 2/4，正样本零误拦截。结论是"知道什么时候不该用它"，比
"加了个新模块"更值得说。

## 8. 会话与缓存

会话默认使用进程内存；Docker Compose 可切换到 Redis，使多个 worker 共享历史。会话保存 user 和 assistant 两种消息，按完整问答轮次裁剪，并设置 TTL 自动过期。

查询改写使用 TTL 缓存。缓存键包含：

```text
原始问题 + 历史消息 + 模型名称 + knowledge_base_version
```

这样知识库或模型版本变化时不会误用旧缓存。当前进程内缓存适合单进程；多实例部署应使用 Redis。

## 9. 可靠性与安全

- `/health/live`：只检查进程是否存活；
- `/health/ready`：检查 VectorStore、Reranker 是否完成初始化；
- Docker healthcheck：依赖 `/health/ready`；
- LLM 调用：超时、有限重试、指数退避；
- 接口：可配置固定窗口限流，超限返回 429；
- 外部依赖：提供熔断器，连续失败后暂时停止调用；
- Prompt：明确 context 是数据而不是指令；
- 输出：过滤空回答和敏感内容；
- 日志：记录 request_id、阶段耗时和数量，不记录 API Key 与完整敏感文档。

## 10. 评测体系

检索层使用人工标注的 `relevant_chunk_ids`：

```text
Recall@K：前 K 个结果是否包含相关 chunk
Precision@K：前 K 个结果中有多少相关 chunk
MRR：第一个相关 chunk 排名的倒数平均值
拒答率：知识库外问题是否返回空 context
```

生成层进一步观察：

```text
Faithfulness：答案是否被 context 支持
Context Recall：context 是否覆盖答案所需信息
Answer Relevance：答案是否真正回答问题
```

当前已完成的检索实验结果：

```text
                         Recall@3  Precision@3  MRR    负样本拒答率
hybrid                   1.000     0.517        0.942  0.000
hybrid + reranker        1.000     0.842        1.000  1.000
```

结论：Reranker 没有改变召回率，但改善了排序和结果纯度，并过滤了知识库外问题。

Router 对比实验也说明：LLM Router 可能把知识库事实误判为常识问题。因此 Router 必须有规则兜底，不能只相信模型决策。

## 11. 测试策略

测试不依赖真实模型：

- FakeStore：模拟向量库；
- FakeReranker：模拟重排；
- fake_llm：验证上下文和历史是否正确传递；
- Fake Redis：验证会话持久化逻辑；
- 异常替身：验证 503、降级、重试、限流和熔断；
- 全量 CI：自动运行 pytest，不发送外部 API 请求。

最近一次本地全量测试：

```text
152 passed, 1 warning
```

## 12. 面试项目串讲模板

> 我独立实现了一个基于 FastAPI 的中文 RAG 知识库问答系统。文档上传后会经过切分、Embedding 和 metadata 构造，向量保存到 Chroma，文档状态保存到 SQLite。检索阶段采用向量检索和 BM25 的混合召回，通过 RRF 融合排名，再使用本地 BGE Reranker 做精排和阈值过滤，最后将可靠 context 交给 DeepSeek 生成答案。系统支持多轮会话、查询改写、SSE 流式输出、Agent Router 和标准 Function Calling Tool Agent。工程上加入了依赖注入、Fake 测试、健康检查、Redis 会话、Docker、超时重试、限流、熔断和 Prometheus 兼容指标。评测方面使用人工标注 chunk_id 计算 Recall、Precision、MRR，并通过消融实验验证 Reranker 对排序质量和负样本拒答的贡献。

## 13. 面试追问的回答顺序

遇到“效果不好怎么办”，按层定位：

```text
先看 Recall → 再看 Precision/MRR → 再看 context 是否支持答案 → 最后看 LLM 生成
```

遇到“为什么这么设计”，按三层回答：

```text
效果：是否提高检索或生成质量
成本：增加了多少模型调用和计算
稳定性：失败时如何超时、降级、拒答和恢复
```

遇到“能不能上线”，必须同时检查：

```text
效果门槛 + P95 延迟 + 错误率 + 成本 + 灰度和回滚能力
```

## 14. 当前项目的边界

这是一个适合展示 RAG 工程能力的个人 Demo，不应夸大为生产级平台。当前还可以继续增强：

- 使用真实 Prometheus client 和 Grafana；
- 对 Redis 限流做多实例原子实现；
- 将文档和会话元数据迁移到 PostgreSQL；
- 增加鉴权、租户隔离和更严格的引用校验；
- 在独立环境重新运行固定 RAG 与 Tool Agent 的 24 题端到端对比实验。
