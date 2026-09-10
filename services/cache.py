"""进程内 TTL 缓存，后续可替换为 Redis 实现。"""

from collections import OrderedDict
from threading import Lock
from time import monotonic
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """线程安全、有最大容量和过期时间的缓存。"""

    def __init__(self, ttl_seconds: float = 300, max_entries: int = 1024):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds 必须大于 0")
        if max_entries <= 0:
            raise ValueError("max_entries 必须大于 0")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._items: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> T | None:
        now = monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._items[key] = (monotonic() + self.ttl_seconds, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
