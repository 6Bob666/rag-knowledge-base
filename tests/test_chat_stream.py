import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.chat import (
    get_llm_stream,
    get_query_rewriter,
    get_reranker,
    router,
)
from services.dependencies import get_store
from services.conversation_store import InMemoryConversationStore
from services.dependencies import get_conversation_store


class FakeStore:
    def hybrid_search_records(self, query, top_k):
        return [
            {
                "id": "chunk-1",
                "text": "CNN 的全称是卷积神经网络。",
                "metadata": {"source": "ai.txt", "chunk_id": "chunk-1"},
            }
        ]


class FakeReranker:
    def rerank_records(self, query, records, top_k):
        return records[:top_k]


def fake_llm_stream(question, context):
    yield "卷"
    yield "积"
    yield "神经网络"


def parse_sse_events(text):
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))
    return events


def test_ask_stream_returns_chunks_in_order():
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    app.dependency_overrides[get_store] = lambda: FakeStore()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: question
    )
    app.dependency_overrides[get_llm_stream] = lambda: fake_llm_stream

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask/stream",
            json={"question": "CNN 的全称是什么？", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: start\n" in response.text
    assert 'event: token\ndata: {"content": "卷"}' in response.text
    assert 'event: token\ndata: {"content": "积"}' in response.text
    assert 'event: token\ndata: {"content": "神经网络"}' in response.text
    done_block = next(
        block for block in response.text.split("\n\n") if block.startswith("event: done")
    )
    assert '"request_id":' in done_block
    assert '"answer": "卷积神经网络"' in done_block
    assert response.text.index("event: start") < response.text.index("event: token")
    assert response.text.index("event: token") < response.text.index("event: done")


def test_ask_stream_events_carry_request_id():
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    app.dependency_overrides[get_store] = lambda: FakeStore()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: question
    )
    app.dependency_overrides[get_llm_stream] = lambda: fake_llm_stream

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask/stream",
            json={
                "conversation_id": "demo-stream-obs",
                "question": "CNN 的全称是什么？",
            },
        )

    events = parse_sse_events(response.text)
    start_data = events[0][1]
    done_data = events[-1][1]
    assert len(start_data["request_id"]) == 12
    assert start_data["conversation_id"] == "demo-stream-obs"
    assert done_data["request_id"] == start_data["request_id"]
    assert events[-1][0] == "done"


def test_ask_stream_rejects_sensitive_output_before_sending_chunks():
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    conversation_store = InMemoryConversationStore()

    def unsafe_llm_stream(question, context):
        yield "正常内容"
        yield " api_key=sk-test-secret-value"

    app.dependency_overrides[get_store] = lambda: FakeStore()
    app.dependency_overrides[get_conversation_store] = lambda: conversation_store
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: question
    )
    app.dependency_overrides[get_llm_stream] = lambda: unsafe_llm_stream

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask/stream",
            json={
                "conversation_id": "stream-unsafe-001",
                "question": "什么是 CNN？",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: error" in response.text
    assert '"code": "INVALID_OUTPUT"' in response.text
    assert "正常内容" not in response.text
    assert conversation_store.get_history("stream-unsafe-001") == []


def test_ask_stream_does_not_save_history_when_generation_fails():
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    conversation_store = InMemoryConversationStore()

    def failing_llm_stream(question, context):
        yield "部分内容"
        raise RuntimeError("模拟流式故障")

    app.dependency_overrides[get_store] = lambda: FakeStore()
    app.dependency_overrides[get_conversation_store] = lambda: conversation_store
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: question
    )
    app.dependency_overrides[get_llm_stream] = lambda: failing_llm_stream

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask/stream",
            json={
                "conversation_id": "stream-failed-001",
                "question": "什么是 CNN？",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: error" in response.text
    assert '"code": "LLM_UNAVAILABLE"' in response.text
    assert "部分内容" not in response.text
    assert conversation_store.get_history("stream-failed-001") == []


def test_ask_stream_returns_error_event_when_context_is_missing():
    from routers.chat import get_query_rewriter

    class EmptyStore:
        def hybrid_search_records(self, query, top_k):
            return []

    app = FastAPI()
    app.include_router(router, prefix="/chat")
    app.dependency_overrides[get_store] = lambda: EmptyStore()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: question
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask/stream",
            json={"question": "没有答案的问题"},
        )

    assert response.status_code == 200
    assert "event: start" in response.text
    assert '"code": "NO_CONTEXT"' in response.text
