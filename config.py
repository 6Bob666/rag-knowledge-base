from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_base_url: str = "https://api.deepseek.com/v1"
    llm_model_name: str = "deepseek-v4-flash"

    database_url: str = f"sqlite:///{PROJECT_ROOT / 'knowledge_base.db'}"
    chroma_path: str = str(PROJECT_ROOT / "chroma_data")
    embedding_model_path: str = str(PROJECT_ROOT / "model" / "bge-small-zh-v1.5")
    reranker_model_path: str = str(PROJECT_ROOT / "model" / "bge-reranker-v2-m3")
    reranker_score_threshold: float = float("-inf")

    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 3

    failure_journal_enabled: bool = False
    failure_journal_path: str = str(
        PROJECT_ROOT / "logs" / "failure_journal.jsonl"
    )

    # Agent Router 模式：rules = 本地规则（零成本、确定性高）；
    # llm = 由大模型输出结构化决策，解析失败时自动回退 rules。
    agent_router_mode: str = "rules"

    query_rewrite_cache_enabled: bool = True
    query_rewrite_cache_ttl_seconds: int = 300
    query_rewrite_cache_max_entries: int = 1024
    knowledge_base_version: str = "local-v1"

    conversation_store_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    conversation_ttl_seconds: int = 86400

    # 长期记忆：默认开启内存版，生产可切到 Redis。
    memory_enabled: bool = True
    memory_store_backend: str = "memory"
    memory_ttl_seconds: int = 2592000
    memory_max_items: int = 200
    memory_recall_top_k: int = 3

    # MCP：开启后 /chat/agent/tool 的工具声明与执行都走 MCP Server 子进程，
    # 而不是在应用进程内直接调用 VectorStore。默认关闭，保持单进程开发简单。
    mcp_enabled: bool = False
    # 自定义启动命令（按空白切分，路径不要带空格）；留空则用
    # "<当前解释器> -m mcp_server.kb_server"。
    mcp_server_command: str = ""
    mcp_request_timeout_seconds: float = 30.0

    # Planner / Verifier：开启后检索循环按"验证结论"选补救策略；
    # 关闭后退回旧行为（只在空结果时重试），出问题可一键切换。
    planner_enabled: bool = True
    verifier_min_coverage: float = 0.5

    # 成本估算单价（元 / 百万 token）。只用于看板量级估算：
    # 不同模型和缓存命中价格差别很大，不能当账单用。
    llm_price_input_per_million: float = 2.0
    llm_price_output_per_million: float = 8.0

    # 稳定性保护；限流默认关闭，避免单机开发和测试受全局状态影响。
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    llm_max_attempts: int = 2
    llm_retry_backoff_seconds: float = 0.2
    llm_timeout_seconds: int = 60
    tool_agent_timeout_seconds: int = 120


settings = Settings()
