"""上传幂等与并发保护的共享工具。

两个问题要分开解决：

**幂等**：同一份内容重复上传，不应该重复切分、重复向量化、重复占用额度。
判断依据是内容哈希，不是文件名——文件名可以重复，内容才是同一份东西。

**并发**：两个请求同时上传同一个文件时，"先查有没有、再决定写不写"这段
临界区会被交错执行，可能双双写入。这里用按文件名的锁把临界区串起来，
不同文件名之间互不阻塞。
"""

import hashlib
import threading
from contextlib import contextmanager


def compute_content_hash(content: bytes) -> str:
    """对文件字节做 SHA-256，作为"这份内容是否已经入库"的判据。"""
    return hashlib.sha256(content).hexdigest()


class KeyedLock:
    """按键加锁：同一个 key 串行，不同 key 并行。"""

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    @contextmanager
    def acquire(self, key: str):
        lock = self._lock_for(key)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def held_keys(self) -> int:
        """当前登记过的 key 数量，便于测试和排查内存增长。"""
        with self._guard:
            return len(self._locks)


upload_lock = KeyedLock()
