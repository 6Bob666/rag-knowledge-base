from threading import Lock

_store = None
_store_lock = Lock()
_conversation_store = None
_conversation_store_lock = Lock()
_memory_store = None
_memory_store_lock = Lock()
_memory_service = None
_memory_service_lock = Lock()
_mcp_client = None
_mcp_client_lock = Lock()
_mcp_provider = None
_mcp_provider_lock = Lock()


def get_store():
    """线程安全地获取共享的 VectorStore 单例。"""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                from services.vector_store import VectorStore
                _store = VectorStore()
    return _store


def get_conversation_store():
    """获取共享的会话历史存储；backend=redis 时使用跨进程存储。"""
    global _conversation_store
    if _conversation_store is None:
        with _conversation_store_lock:
            if _conversation_store is None:
                from config import settings
                from services.conversation_store import (
                    InMemoryConversationStore,
                    RedisConversationStore,
                )

                if settings.conversation_store_backend.lower() == "redis":
                    _conversation_store = RedisConversationStore(
                        settings.redis_url,
                        ttl_seconds=settings.conversation_ttl_seconds,
                    )
                else:
                    _conversation_store = InMemoryConversationStore()
    return _conversation_store


def get_memory_store():
    """获取长期记忆存储；默认内存，多实例部署可切换 Redis。"""
    global _memory_store
    if _memory_store is None:
        with _memory_store_lock:
            if _memory_store is None:
                from config import settings
                from services.memory_store import (
                    InMemoryMemoryStore,
                    RedisMemoryStore,
                )

                if settings.memory_store_backend.lower() == "redis":
                    _memory_store = RedisMemoryStore(
                        settings.redis_url,
                        max_items_per_user=settings.memory_max_items,
                        ttl_seconds=settings.memory_ttl_seconds,
                    )
                else:
                    _memory_store = InMemoryMemoryStore(
                        max_items_per_user=settings.memory_max_items
                    )
    return _memory_store


def get_memory_service():
    """获取长期记忆服务；关闭记忆时返回 None。"""
    global _memory_service
    from config import settings

    if not settings.memory_enabled:
        return None
    if _memory_service is None:
        with _memory_service_lock:
            if _memory_service is None:
                from services.memory_service import MemoryService

                _memory_service = MemoryService(
                    get_memory_store(),
                    recall_top_k=settings.memory_recall_top_k,
                )
    return _memory_service


def get_mcp_client():
    """获取 MCP Server 子进程客户端单例；未启用 MCP 时返回 None。

    子进程只在第一次使用时启动（懒加载），避免不用 Agent 接口的部署
    也白白常驻一份进程和内存。
    """
    global _mcp_client
    from config import settings

    if not settings.mcp_enabled:
        return None
    if _mcp_client is None:
        with _mcp_client_lock:
            if _mcp_client is None:
                from services.mcp_client import StdioMCPClient

                command = (settings.mcp_server_command or "").split() or None
                client = StdioMCPClient(
                    command,
                    request_timeout=settings.mcp_request_timeout_seconds,
                )
                client.start()
                client.initialize()
                _mcp_client = client
    return _mcp_client


def get_mcp_tool_provider():
    """把 MCP 客户端包装成 Agent 工具源；未启用 MCP 时返回 None。"""
    global _mcp_provider
    client = get_mcp_client()
    if client is None:
        return None
    if _mcp_provider is None:
        with _mcp_provider_lock:
            if _mcp_provider is None:
                from services.mcp_tools import MCPToolProvider

                _mcp_provider = MCPToolProvider(client)
    return _mcp_provider


def shutdown_mcp_client() -> None:
    """应用关闭时释放 MCP 子进程，避免留下孤儿进程。"""
    global _mcp_client, _mcp_provider
    with _mcp_client_lock:
        if _mcp_client is not None:
            try:
                _mcp_client.close()
            except Exception:
                pass
            _mcp_client = None
    _mcp_provider = None
