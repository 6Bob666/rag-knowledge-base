import json

import pytest

from services.conversation_store import (
    ConversationStoreError,
    RedisConversationStore,
)


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.pending = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set(self, key, value, ex):
        self.pending = (key, value, ex)

    def execute(self):
        key, value, ex = self.pending
        self.client.values[key] = value
        self.client.expirations[key] = ex
        return [True]


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = {}

    def get(self, key):
        return self.values.get(key)

    def pipeline(self):
        return FakePipeline(self)

    def delete(self, key):
        self.values.pop(key, None)


def test_redis_store_reads_writes_and_trims_history():
    client = FakeRedis()
    store = RedisConversationStore(
        "redis://fake",
        max_messages=2,
        ttl_seconds=60,
        client=client,
    )
    store.append_turn("c1", "问题 1", "回答 1")
    store.append_turn("c1", "问题 2", "回答 2")
    assert store.get_history("c1") == [
        {"role": "user", "content": "问题 2"},
        {"role": "assistant", "content": "回答 2"},
    ]
    assert client.expirations["rag:conversation:c1"] == 60


def test_redis_store_clear_and_empty_conversation():
    client = FakeRedis()
    store = RedisConversationStore("redis://fake", client=client)
    assert store.get_history(None) == []
    store.append_turn("c1", "问题", "回答")
    store.clear("c1")
    assert store.get_history("c1") == []


def test_redis_store_does_not_hide_connection_error():
    class BrokenRedis:
        def get(self, key):
            raise ConnectionError("redis down")

    store = RedisConversationStore("redis://fake", client=BrokenRedis())
    with pytest.raises(ConversationStoreError, match="读取失败"):
        store.get_history("c1")


def test_redis_store_rejects_corrupted_payload():
    client = FakeRedis()
    client.values["rag:conversation:c1"] = json.dumps({"not": "a list"})
    store = RedisConversationStore("redis://fake", client=client)
    with pytest.raises(ConversationStoreError, match="必须是数组"):
        store.get_history("c1")
