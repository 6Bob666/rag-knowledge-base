import time

from services.cache import TTLCache
from services.query_rewrite_cache import CachedQueryRewriter


def test_ttl_cache_get_set_and_delete():
    cache = TTLCache[str](ttl_seconds=1, max_entries=2)
    assert cache.get("missing") is None
    cache.set("a", "A")
    assert cache.get("a") == "A"
    cache.delete("a")
    assert cache.get("a") is None


def test_ttl_cache_expires():
    cache = TTLCache[str](ttl_seconds=0.01, max_entries=2)
    cache.set("a", "A")
    time.sleep(0.02)
    assert cache.get("a") is None


def test_ttl_cache_evicts_oldest_entry_when_full():
    cache = TTLCache[str](ttl_seconds=10, max_entries=2)
    cache.set("a", "A")
    cache.set("b", "B")
    cache.set("c", "C")
    assert cache.get("a") is None
    assert cache.get("b") == "B"
    assert cache.get("c") == "C"


def test_cached_rewriter_uses_question_and_history_as_key():
    calls = []

    def fake_rewriter(question, history):
        calls.append((question, history))
        return f"改写：{question}"

    rewriter = CachedQueryRewriter(
        fake_rewriter,
        cache=TTLCache[str](ttl_seconds=10, max_entries=10),
        model_name="fake-model",
        knowledge_base_version="kb-v1",
    )
    history = [{"role": "user", "content": "什么是 CNN？"}]
    assert rewriter("它有什么应用？", history) == "改写：它有什么应用？"
    assert rewriter("它有什么应用？", history) == "改写：它有什么应用？"
    assert len(calls) == 1
    changed_history = history + [
        {"role": "assistant", "content": "CNN 是卷积神经网络。"}
    ]
    rewriter("它有什么应用？", changed_history)
    assert len(calls) == 2


def test_cached_rewriter_falls_back_when_rewriter_returns_empty():
    rewriter = CachedQueryRewriter(
        lambda question, history: "   ",
        cache=TTLCache[str](ttl_seconds=10, max_entries=10),
        model_name="fake-model",
        knowledge_base_version="kb-v1",
    )
    assert rewriter("原问题", []) == "原问题"


def test_cache_key_changes_when_knowledge_base_version_changes():
    first = CachedQueryRewriter(
        lambda question, history: question,
        cache=TTLCache[str](ttl_seconds=10, max_entries=10),
        model_name="fake-model",
        knowledge_base_version="kb-v1",
    )
    second = CachedQueryRewriter(
        lambda question, history: question,
        cache=first.cache,
        model_name="fake-model",
        knowledge_base_version="kb-v2",
    )
    assert first._key("q", []) != second._key("q", [])
