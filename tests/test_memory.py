from services.memory_extractor import RuleBasedMemoryExtractor
from services.memory_service import MemoryService
from services.memory_store import InMemoryMemoryStore, RedisMemoryStore


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        self.values[key] = value
        self.expirations[key] = ex

    def delete(self, key):
        self.values.pop(key, None)


def test_inmemory_store_add_search_and_dedupe():
    store = InMemoryMemoryStore()
    store.add("u1", "用户关注：RAG 检索")
    store.add("u1", "用户关注：RAG 检索")
    store.add("u1", "用户偏好：答案简洁")

    assert len(store.list("u1")) == 2
    hits = store.search("u1", "RAG 检索怎么优化？", top_k=2)
    assert hits
    assert "RAG" in hits[0].record.content
    assert store.search("u2", "RAG", top_k=2) == []


def test_inmemory_store_clear():
    store = InMemoryMemoryStore()
    store.add("u1", "用户自称：张三")
    store.clear("u1")
    assert store.list("u1") == []


def test_redis_memory_store_roundtrip_and_ttl():
    client = FakeRedis()
    store = RedisMemoryStore("redis://fake", client=client, ttl_seconds=60)
    store.add("u1", "用户关注：Agent 编排")
    hits = store.search("u1", "Agent", top_k=1)
    assert hits[0].record.content == "用户关注：Agent 编排"
    assert client.expirations["rag:memory:u1"] == 60
    store.clear("u1")
    assert store.list("u1") == []


def test_rule_extractor_finds_identity_and_preference():
    extractor = RuleBasedMemoryExtractor()
    candidates = extractor.extract("我叫刘永超，我关注 RAG 检索，以后请用中文回答。")
    contents = {candidate.content for candidate in candidates}
    assert "用户自称：刘永超" in contents
    assert "用户关注：RAG" in contents
    assert any("用中文回答" in content for content in contents)


def test_rule_extractor_ignores_plain_question():
    assert RuleBasedMemoryExtractor().extract("CNN 的全称是什么？") == []


def test_memory_service_remembers_and_formats():
    service = MemoryService(InMemoryMemoryStore(), recall_top_k=3)
    records = service.remember_turn("u1", "我叫刘永超，我研究 RAG。")
    assert len(records) == 2

    recalled = service.recall("u1", "RAG 怎么做？")
    text = service.format_for_prompt(recalled)
    assert "<user_memory>" in text
    assert "RAG" in text
    assert service.format_for_prompt([]) == ""
