"""长期记忆存储。

记忆不是知识库事实，而是用于个性化的用户信息，例如用户身份、
关注方向、偏好和明确要求记住的内容。
"""

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock


class MemoryStoreError(RuntimeError):
    """记忆存储不可用或数据格式异常。"""


@dataclass
class MemoryRecord:
    memory_id: str
    user_id: str
    content: str
    kind: str = "fact"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryRecord":
        return cls(
            memory_id=str(data.get("memory_id") or uuid.uuid4().hex),
            user_id=str(data.get("user_id") or ""),
            content=str(data.get("content") or ""),
            kind=str(data.get("kind") or "fact"),
            created_at=str(data.get("created_at") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class MemoryHit:
    record: MemoryRecord
    score: float


def _tokenize(text: str) -> list[str]:
    """中英文混合的分词：英文按词，中文按单字和二元组。"""
    lowered = (text or "").lower()
    latin = re.findall(r"[a-z0-9_]+", lowered)
    han = re.findall(r"[\u4e00-\u9fff]", lowered)
    bigrams = ["".join(pair) for pair in zip(han, han[1:])]
    return latin + han + bigrams


def _recency_bonus(created_at: str, now: datetime | None = None) -> float:
    if not created_at:
        return 0.0
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    days = max(0.0, (current - created).total_seconds() / 86400)
    return 1.0 / (1.0 + days / 30.0)


def score_memory(query: str, record: MemoryRecord) -> float:
    """关键词覆盖率 + Jaccard 相似度 + 时间新鲜度。"""
    query_tokens = set(_tokenize(query))
    memory_tokens = set(_tokenize(record.content))
    if not query_tokens or not memory_tokens:
        return 0.0
    overlap = query_tokens & memory_tokens
    if not overlap:
        return 0.0
    coverage = len(overlap) / len(query_tokens)
    jaccard = len(overlap) / len(query_tokens | memory_tokens)
    return round(0.6 * coverage + 0.4 * jaccard + 0.1 * _recency_bonus(record.created_at), 6)


class InMemoryMemoryStore:
    """线程安全的内存记忆存储，适合单进程和测试。"""

    def __init__(self, max_items_per_user: int = 200):
        if max_items_per_user <= 0:
            raise ValueError("max_items_per_user 必须大于 0")
        self.max_items_per_user = max_items_per_user
        self._items: dict[str, list[MemoryRecord]] = {}
        self._lock = Lock()

    def add(
        self,
        user_id: str,
        content: str,
        kind: str = "fact",
        metadata: dict | None = None,
    ) -> MemoryRecord:
        if not user_id:
            raise ValueError("user_id 不能为空")
        normalized = (content or "").strip()
        if not normalized:
            raise ValueError("记忆内容不能为空")

        with self._lock:
            items = self._items.setdefault(user_id, [])
            for existing in items:
                if existing.content == normalized:
                    existing.created_at = datetime.now(timezone.utc).isoformat()
                    return existing
            record = MemoryRecord(
                memory_id=uuid.uuid4().hex,
                user_id=user_id,
                content=normalized,
                kind=kind,
                metadata=dict(metadata or {}),
            )
            items.append(record)
            if len(items) > self.max_items_per_user:
                del items[: len(items) - self.max_items_per_user]
            return record

    def search(self, user_id: str, query: str, top_k: int = 3) -> list[MemoryHit]:
        if not user_id or top_k <= 0:
            return []
        with self._lock:
            items = list(self._items.get(user_id, []))
        hits = [
            MemoryHit(record=record, score=score_memory(query, record))
            for record in items
        ]
        hits = [hit for hit in hits if hit.score > 0]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def list(self, user_id: str) -> list[MemoryRecord]:
        if not user_id:
            return []
        with self._lock:
            return list(self._items.get(user_id, []))

    def clear(self, user_id: str) -> None:
        if not user_id:
            return
        with self._lock:
            self._items.pop(user_id, None)


class RedisMemoryStore:
    """基于 Redis 的共享长期记忆存储。"""

    def __init__(
        self,
        redis_url: str,
        *,
        max_items_per_user: int = 200,
        ttl_seconds: int = 2592000,
        client=None,
    ):
        if max_items_per_user <= 0 or ttl_seconds <= 0:
            raise ValueError("记忆存储参数必须大于 0")
        if client is None:
            try:
                import redis
            except ImportError as exc:
                raise MemoryStoreError("Redis 记忆存储需要安装 redis 依赖") from exc
            client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.client = client
        self.max_items_per_user = max_items_per_user
        self.ttl_seconds = ttl_seconds
        self.key_prefix = "rag:memory:"

    def _key(self, user_id: str) -> str:
        return f"{self.key_prefix}{user_id}"

    def _load(self, user_id: str) -> list[MemoryRecord]:
        try:
            raw = self.client.get(self._key(user_id))
        except Exception as exc:
            raise MemoryStoreError("Redis 记忆读取失败") from exc
        if raw is None:
            return []
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise MemoryStoreError("Redis 记忆数据格式无效") from exc
        if not isinstance(data, list):
            raise MemoryStoreError("Redis 记忆数据必须是数组")
        return [MemoryRecord.from_dict(item) for item in data]

    def _save(self, user_id: str, records: list[MemoryRecord]) -> None:
        try:
            self.client.set(
                self._key(user_id),
                json.dumps([record.to_dict() for record in records], ensure_ascii=False),
                ex=self.ttl_seconds,
            )
        except Exception as exc:
            raise MemoryStoreError("Redis 记忆写入失败") from exc

    def add(
        self,
        user_id: str,
        content: str,
        kind: str = "fact",
        metadata: dict | None = None,
    ) -> MemoryRecord:
        if not user_id:
            raise ValueError("user_id 不能为空")
        normalized = (content or "").strip()
        if not normalized:
            raise ValueError("记忆内容不能为空")
        records = self._load(user_id)
        for existing in records:
            if existing.content == normalized:
                existing.created_at = datetime.now(timezone.utc).isoformat()
                self._save(user_id, records)
                return existing
        record = MemoryRecord(
            memory_id=uuid.uuid4().hex,
            user_id=user_id,
            content=normalized,
            kind=kind,
            metadata=dict(metadata or {}),
        )
        records.append(record)
        records = records[-self.max_items_per_user :]
        self._save(user_id, records)
        return record

    def search(self, user_id: str, query: str, top_k: int = 3) -> list[MemoryHit]:
        if not user_id or top_k <= 0:
            return []
        hits = [
            MemoryHit(record=record, score=score_memory(query, record))
            for record in self._load(user_id)
        ]
        hits = [hit for hit in hits if hit.score > 0]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def list(self, user_id: str) -> list[MemoryRecord]:
        return self._load(user_id) if user_id else []

    def clear(self, user_id: str) -> None:
        if not user_id:
            return
        try:
            self.client.delete(self._key(user_id))
        except Exception as exc:
            raise MemoryStoreError("Redis 记忆删除失败") from exc
