from threading import Lock

_store = None
_store_lock = Lock()
_conversation_store = None
_conversation_store_lock = Lock()


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
