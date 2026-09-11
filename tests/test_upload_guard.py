"""上传并发保护与内容哈希的测试。"""

import threading
import time

import pytest

from services.upload_guard import KeyedLock, compute_content_hash


def test_content_hash_is_stable_and_content_sensitive():
    assert compute_content_hash(b"abc") == compute_content_hash(b"abc")
    assert compute_content_hash(b"abc") != compute_content_hash(b"abd")
    assert len(compute_content_hash(b"abc")) == 64


def test_keyed_lock_serialises_same_key():
    lock = KeyedLock()
    events: list[str] = []

    def worker(name: str):
        with lock.acquire("same.txt"):
            events.append(f"{name}-in")
            time.sleep(0.05)
            events.append(f"{name}-out")

    threads = [
        threading.Thread(target=worker, args=(f"t{index}",))
        for index in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # 同一个 key 必须一个进一个出，不能交错。
    for index in range(0, len(events), 2):
        assert events[index].endswith("-in")
        assert events[index + 1].endswith("-out")
        assert events[index].split("-")[0] == events[index + 1].split("-")[0]


def test_keyed_lock_allows_different_keys_to_run_concurrently():
    lock = KeyedLock()
    barrier = threading.Barrier(2, timeout=5)

    def worker(key: str):
        with lock.acquire(key):
            # 两个不同的 key 能同时进到临界区，说明没有全局串行化。
            barrier.wait()

    threads = [
        threading.Thread(target=worker, args=(key,))
        for key in ("a.txt", "b.txt")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(not thread.is_alive() for thread in threads)


def test_keyed_lock_releases_on_exception():
    lock = KeyedLock()

    with pytest.raises(RuntimeError):
        with lock.acquire("a.txt"):
            raise RuntimeError("boom")

    # 异常后锁必须释放，否则这个文件名会被永久卡死。
    acquired = threading.Event()

    def worker():
        with lock.acquire("a.txt"):
            acquired.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=2)
    assert acquired.is_set()
