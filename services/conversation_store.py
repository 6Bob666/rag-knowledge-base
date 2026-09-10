"""会话历史存储。

当前版本使用进程内存保存历史，目的是先把多轮对话的数据流跑通。
生产环境可以在不修改路由业务逻辑的前提下替换为 Redis 或数据库实现。
"""

from copy import deepcopy
import json
from threading import Lock


class ConversationStoreError(RuntimeError):
    """会话存储不可用或数据格式异常。"""


class InMemoryConversationStore:
    """线程安全的简单会话存储。"""

    def __init__(self, max_messages: int = 12):
        if max_messages < 2:
            raise ValueError("max_messages 必须至少为 2")
        self.max_messages = max_messages
        self._conversations: dict[str, list[dict[str, str]]] = {}
        self._lock = Lock()

    def get_history(self, conversation_id: str | None) -> list[dict[str, str]]:
        """返回历史副本，避免调用方直接修改存储内容。"""
        if not conversation_id:
            return []
        with self._lock:
            return deepcopy(self._conversations.get(conversation_id, []))

    def append_turn(
        self,
        conversation_id: str | None,
        question: str,
        answer: str,
    ) -> None:
        """保存一轮成功问答；没有会话 ID 时保持单轮无状态。"""
        if not conversation_id or not question.strip() or not answer.strip():
            return

        with self._lock:
            history = self._conversations.setdefault(conversation_id, [])
            history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
            self._conversations[conversation_id] = history[-self.max_messages :]

    def clear(self, conversation_id: str | None) -> None:
        """删除指定会话，供测试和后续会话管理接口使用。"""
        if not conversation_id:
            return
        with self._lock:
            self._conversations.pop(conversation_id, None)


class RedisConversationStore:
    """基于 Redis 的共享会话存储。

    Redis 不可用时直接抛出异常，由接口层返回服务不可用，
    不把故障伪装成空历史。
    """

    def __init__(
        self,
        redis_url: str,
        max_messages: int = 12,
        ttl_seconds: int = 86400,
        *,
        client=None,
    ):
        if max_messages < 2:
            raise ValueError("max_messages 必须至少为 2")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds 必须大于 0")
        if client is None:
            try:
                import redis
            except ImportError as exc:
                raise ConversationStoreError(
                    "Redis 会话存储需要安装 redis 依赖"
                ) from exc
            client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.client = client
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self.key_prefix = "rag:conversation:"

    def _key(self, conversation_id: str) -> str:
        return f"{self.key_prefix}{conversation_id}"

    @staticmethod
    def _decode(raw) -> list[dict[str, str]]:
        if raw is None:
            return []
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ConversationStoreError("Redis 会话数据格式无效") from exc
        if not isinstance(value, list):
            raise ConversationStoreError("Redis 会话数据必须是数组")
        return value

    def get_history(self, conversation_id: str | None) -> list[dict[str, str]]:
        if not conversation_id:
            return []
        try:
            return deepcopy(self._decode(self.client.get(self._key(conversation_id))))
        except ConversationStoreError:
            raise
        except Exception as exc:
            raise ConversationStoreError("Redis 会话读取失败") from exc

    def append_turn(
        self,
        conversation_id: str | None,
        question: str,
        answer: str,
    ) -> None:
        if not conversation_id or not question.strip() or not answer.strip():
            return
        key = self._key(conversation_id)
        try:
            # pipeline 保证读取、裁剪、写回在一次事务中完成。
            with self.client.pipeline() as pipe:
                raw = self.client.get(key)
                history = self._decode(raw)
                history.extend(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                )
                pipe.set(
                    key,
                    json.dumps(history[-self.max_messages :], ensure_ascii=False),
                    ex=self.ttl_seconds,
                )
                pipe.execute()
        except ConversationStoreError:
            raise
        except Exception as exc:
            raise ConversationStoreError("Redis 会话写入失败") from exc

    def clear(self, conversation_id: str | None) -> None:
        if not conversation_id:
            return
        try:
            self.client.delete(self._key(conversation_id))
        except Exception as exc:
            raise ConversationStoreError("Redis 会话删除失败") from exc
