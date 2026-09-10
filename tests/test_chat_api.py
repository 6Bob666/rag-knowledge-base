from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.chat import (
    get_llm,
    get_query_rewriter,
    get_reranker,
    router,
)
from services.dependencies import get_store
from services.conversation_store import InMemoryConversationStore
from services.dependencies import get_conversation_store


class FakeStore:
    def __init__(self):
        self.calls = []

    def hybrid_search_records(self, query, top_k):
        self.calls.append((query, top_k))
        return [
            {
                "id": "chunk-1",
                "text": "CNN 的全称是 Convolutional Neural Network。",
                "metadata": {"source": "ai.txt", "chunk_id": "chunk-1"},
            },
            {
                "id": "chunk-2",
                "text": "CNN 是一种深度学习模型。",
                "metadata": {"source": "ai.txt", "chunk_id": "chunk-2"},
            },
        ]


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank_records(self, query, records, top_k):
        self.calls.append((query, len(records), top_k))
        return records[:top_k]


class EmptyReranker:
    def rerank_records(self, query, records, top_k):
        return []


class EmptyStore:
    def hybrid_search_records(self, query, top_k):
        return []


def test_ask_uses_injected_dependencies():
    app = FastAPI()
    app.include_router(router, prefix="/chat")

    fake_store = FakeStore()
    fake_reranker = FakeReranker()
    app.dependency_overrides[get_store] = lambda: fake_store
    app.dependency_overrides[get_reranker] = lambda: fake_reranker
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: f"改写后：{question}"
    )
    app.dependency_overrides[get_llm] = lambda: (
        lambda question, context: "这是测试模型生成的回答"
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "CNN 的全称是什么？", "top_k": 1},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "这是测试模型生成的回答",
        "contexts": [
            {
                "text": "CNN 的全称是 Convolutional Neural Network。",
                "source": "ai.txt",
                "chunk_id": "chunk-1",
            }
        ],
    }
    assert fake_store.calls == [("改写后：CNN 的全称是什么？", 3)]
    assert fake_reranker.calls == [("改写后：CNN 的全称是什么？", 2, 1)]


def test_ask_returns_empty_context_when_nothing_is_retrieved():
    app = FastAPI()
    app.include_router(router, prefix="/chat")

    llm_calls = []

    def fake_llm(question, context):
        llm_calls.append((question, context))
        return "不应该被调用"

    app.dependency_overrides[get_store] = lambda: EmptyStore()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: question
    )
    app.dependency_overrides[get_llm] = lambda: fake_llm

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "一个没有答案的问题", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "未找到相关文档。",
        "contexts": [],
    }
    assert llm_calls == []


def test_ask_returns_empty_context_when_all_candidates_are_low_score():
    app = FastAPI()
    app.include_router(router, prefix="/chat")

    llm_calls = []

    def fake_llm(question, context):
        llm_calls.append((question, context))
        return "不应该被调用"

    app.dependency_overrides[get_store] = lambda: FakeStore()
    app.dependency_overrides[get_reranker] = lambda: EmptyReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: question
    )
    app.dependency_overrides[get_llm] = lambda: fake_llm

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "一个没有足够相关内容的问题", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "未找到足够相关的文档。",
        "contexts": [],
    }
    assert llm_calls == []


def test_ask_reuses_history_for_same_conversation():
    app = FastAPI()
    app.include_router(router, prefix="/chat")

    fake_store = FakeStore()
    conversation_store = InMemoryConversationStore()
    rewriter_calls = []
    llm_calls = []

    def fake_rewriter(question, history):
        rewriter_calls.append((question, history))
        if history:
            return "CNN 有哪些应用？"
        return question

    def fake_llm(question, context):
        llm_calls.append((question, context))
        return f"回答第 {len(llm_calls)} 轮"

    app.dependency_overrides[get_store] = lambda: fake_store
    app.dependency_overrides[get_conversation_store] = lambda: conversation_store
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: fake_rewriter
    app.dependency_overrides[get_llm] = lambda: fake_llm

    with TestClient(app) as client:
        first = client.post(
            "/chat/ask",
            json={
                "conversation_id": "demo-001",
                "question": "什么是 CNN？",
            },
        )
        second = client.post(
            "/chat/ask",
            json={
                "conversation_id": "demo-001",
                "question": "它有哪些应用？",
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert rewriter_calls[0] == ("什么是 CNN？", [])
    assert rewriter_calls[1][0] == "它有哪些应用？"
    assert rewriter_calls[1][1] == [
        {"role": "user", "content": "什么是 CNN？"},
        {"role": "assistant", "content": "回答第 1 轮"},
    ]
    assert fake_store.calls[1][0] == "CNN 有哪些应用？"
    assert conversation_store.get_history("demo-001")[-2:] == [
        {"role": "user", "content": "它有哪些应用？"},
        {"role": "assistant", "content": "回答第 2 轮"},
    ]


def test_ask_does_not_save_history_when_llm_fails():
    app = FastAPI()
    app.include_router(router, prefix="/chat")

    conversation_store = InMemoryConversationStore()

    def failing_llm(question, context):
        raise RuntimeError("模拟 LLM 故障")

    app.dependency_overrides[get_store] = lambda: FakeStore()
    app.dependency_overrides[get_conversation_store] = lambda: conversation_store
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: question
    )
    app.dependency_overrides[get_llm] = lambda: failing_llm

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={
                "conversation_id": "failed-001",
                "question": "什么是 CNN？",
            },
        )

    assert response.status_code == 503
    assert conversation_store.get_history("failed-001") == []


def test_ask_passes_history_to_three_argument_llm():
    app = FastAPI()
    app.include_router(router, prefix="/chat")

    conversation_store = InMemoryConversationStore()
    conversation_store.append_turn(
        "demo-llm-001",
        "什么是 CNN？",
        "CNN 是卷积神经网络。",
    )
    llm_calls = []

    def fake_llm(question, context, history):
        llm_calls.append((question, context, history))
        return "CNN 的相关回答"

    app.dependency_overrides[get_store] = lambda: FakeStore()
    app.dependency_overrides[get_conversation_store] = lambda: conversation_store
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question, history: "CNN 有哪些应用？"
    )
    app.dependency_overrides[get_llm] = lambda: fake_llm

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={
                "conversation_id": "demo-llm-001",
                "question": "它有哪些应用？",
            },
        )

    assert response.status_code == 200
    assert len(llm_calls) == 1
    assert llm_calls[0][0] == "它有哪些应用？"
    assert "CNN 的全称是" in llm_calls[0][1]
    assert llm_calls[0][2] == [
        {"role": "user", "content": "什么是 CNN？"},
        {"role": "assistant", "content": "CNN 是卷积神经网络。"},
    ]


def test_ask_rejects_empty_llm_output_and_does_not_save_history():
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    conversation_store = InMemoryConversationStore()

    app.dependency_overrides[get_store] = lambda: FakeStore()
    app.dependency_overrides[get_conversation_store] = lambda: conversation_store
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: question
    )
    app.dependency_overrides[get_llm] = lambda: (lambda question, context: "   ")

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"conversation_id": "empty-001", "question": "什么是 CNN？"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "问答模型返回了无效内容"
    assert conversation_store.get_history("empty-001") == []


def test_ask_rejects_sensitive_llm_output_and_does_not_save_history():
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    conversation_store = InMemoryConversationStore()

    app.dependency_overrides[get_store] = lambda: FakeStore()
    app.dependency_overrides[get_conversation_store] = lambda: conversation_store
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: question
    )
    app.dependency_overrides[get_llm] = lambda: (
        lambda question, context: "api_key=sk-test-secret-value"
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"conversation_id": "secret-001", "question": "什么是 CNN？"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "问答模型返回了无效内容"
    assert conversation_store.get_history("secret-001") == []


class RetryStore:
    def __init__(self):
        self.calls = []

    def hybrid_search_records(self, query, top_k):
        self.calls.append(query)
        if query.startswith("改写后"):
            return []
        return [
            {
                "id": "chunk-retry",
                "text": "原问题检索到的补充资料。",
                "metadata": {"source": "ai.txt", "chunk_id": "chunk-retry"},
            }
        ]


class RecordingEmptyStore:
    def __init__(self):
        self.calls = []

    def hybrid_search_records(self, query, top_k):
        self.calls.append(query)
        return []


def test_ask_retries_with_original_query_when_rewritten_query_has_no_context():
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    fake_store = RetryStore()

    app.dependency_overrides[get_store] = lambda: fake_store
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: f"改写后：{question}"
    )
    app.dependency_overrides[get_llm] = lambda: (
        lambda question, context: "补救后成功回答"
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "补救问题", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "补救后成功回答"
    assert fake_store.calls == ["改写后：补救问题", "补救问题"]


def test_ask_retry_is_bounded_when_original_query_also_has_no_context():
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    fake_store = RecordingEmptyStore()
    llm_calls = []

    app.dependency_overrides[get_store] = lambda: fake_store
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: f"改写后：{question}"
    )
    app.dependency_overrides[get_llm] = lambda: (
        lambda question, context: llm_calls.append((question, context)) or "不应被调用"
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "完全不在知识库的问题", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "未找到相关文档。"
    assert fake_store.calls == [
        "改写后：完全不在知识库的问题",
        "完全不在知识库的问题",
    ]
    assert llm_calls == []


def test_ask_skips_retry_when_enable_retry_is_false():
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    fake_store = RetryStore()
    llm_calls = []

    app.dependency_overrides[get_store] = lambda: fake_store
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: f"改写后：{question}"
    )
    app.dependency_overrides[get_llm] = lambda: (
        lambda question, context: llm_calls.append((question, context)) or "不应被调用"
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={
                "question": "补救问题",
                "enable_retry": False,
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "未找到相关文档。"
    assert fake_store.calls == ["改写后：补救问题"]
    assert llm_calls == []
