import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.chat import (
    get_agent_router,
    get_llm,
    get_llm_stream,
    get_query_rewriter,
    get_reranker,
    router,
)
from services.agent_state import RouteDecision
from services.conversation_store import InMemoryConversationStore
from services.dependencies import get_conversation_store, get_store


class FakeStore:
    def __init__(self):
        self.calls = []

    def hybrid_search_records(self, query, top_k):
        self.calls.append((query, top_k))
        return [
            {
                "id": "chunk-1",
                "text": "CNN 是一种深度学习模型。",
                "metadata": {"source": "ai.txt", "chunk_id": "chunk-1"},
            }
        ]


class ExplodingStore:
    def hybrid_search_records(self, query, top_k):
        raise AssertionError("此分支不应该触发检索")


class FakeReranker:
    def rerank_records(self, query, records, top_k):
        return records[:top_k]


def fail_rewriter(question):
    raise AssertionError("此分支不应调用查询改写")


def fail_llm(question, context):
    raise AssertionError("此分支不应调用 LLM")


def fail_router(question):
    raise RuntimeError("Router 模型故障")


def build_app(
    *,
    store,
    reranker,
    query_rewriter,
    llm,
    agent_router,
    conversation_store=None,
):
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    if conversation_store is None:
        conversation_store = InMemoryConversationStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_reranker] = lambda: reranker
    app.dependency_overrides[get_query_rewriter] = lambda: query_rewriter
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_agent_router] = lambda: agent_router
    app.dependency_overrides[
        get_conversation_store
    ] = lambda: conversation_store
    return app


def test_direct_answer_skips_retrieval():
    app = build_app(
        store=ExplodingStore(),
        reranker=FakeReranker(),
        query_rewriter=fail_rewriter,
        llm=lambda question, context: "你好！有什么可以帮你？",
        agent_router=lambda question: RouteDecision(
            action="direct_answer",
            reason="闲聊",
            confidence=0.9,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "你好"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "你好！有什么可以帮你？",
        "contexts": [],
    }


def test_refuse_returns_answer_without_calling_tools_or_llm():
    app = build_app(
        store=ExplodingStore(),
        reranker=FakeReranker(),
        query_rewriter=fail_rewriter,
        llm=fail_llm,
        agent_router=lambda question: RouteDecision(
            action="refuse",
            reason="超出能力范围",
            confidence=0.95,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "帮我把服务器重启一下"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "抱歉，我暂时无法回答这个问题。",
        "contexts": [],
    }


def test_router_failure_safely_falls_back_to_retrieve():
    fake_store = FakeStore()
    app = build_app(
        store=fake_store,
        reranker=FakeReranker(),
        query_rewriter=lambda question: question,
        llm=lambda question, context: "Router 失败后仍能正常回答",
        agent_router=fail_router,
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "CNN 是什么？", "top_k": 1},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Router 失败后仍能正常回答"
    assert fake_store.calls == [("CNN 是什么？", 3)]


def test_router_query_skips_duplicate_rewrite():
    fake_store = FakeStore()
    llm_calls = []
    app = build_app(
        store=fake_store,
        reranker=FakeReranker(),
        query_rewriter=fail_rewriter,
        llm=lambda question, context: llm_calls.append((question, context))
        or "带自定义检索问法的回答",
        agent_router=lambda question: RouteDecision(
            action="retrieve",
            query="CNN 有哪些典型应用？",
            reason="Router 直接给出检索问法",
            confidence=0.85,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "CNN 的应用", "top_k": 1},
        )

    assert response.status_code == 200
    assert fake_store.calls == [("CNN 有哪些典型应用？", 3)]
    assert llm_calls[0][0] == "CNN 的应用"
    assert "CNN 是一种深度学习模型。" in llm_calls[0][1]


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


def test_stream_direct_answer_skips_retrieval():
    app = FastAPI()
    app.include_router(router, prefix="/chat")

    def fake_llm_stream(question, context):
        assert context == ""
        yield "你"
        yield "好"

    app.dependency_overrides[get_store] = lambda: ExplodingStore()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: fail_rewriter
    app.dependency_overrides[get_llm_stream] = lambda: fake_llm_stream
    app.dependency_overrides[get_agent_router] = lambda: (
        lambda question: RouteDecision(
            action="direct_answer",
            reason="闲聊",
            confidence=0.9,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask/stream",
            json={"question": "你好"},
        )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[0][0] == "start"
    assert [(event, data) for event, data in events if event == "token"] == [
        ("token", {"content": "你"}),
        ("token", {"content": "好"}),
    ]
    assert events[-1][0] == "done"
    assert events[-1][1]["answer"] == "你好"


def test_stream_refuse_returns_error_without_calling_tools_or_llm():
    app = FastAPI()
    app.include_router(router, prefix="/chat")

    def fake_llm_stream(question, context):
        raise AssertionError("refuse 不应调用流式 LLM")

    app.dependency_overrides[get_store] = lambda: ExplodingStore()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: fail_rewriter
    app.dependency_overrides[get_llm_stream] = lambda: fake_llm_stream
    app.dependency_overrides[get_agent_router] = lambda: (
        lambda question: RouteDecision(
            action="refuse",
            reason="超出能力范围",
            confidence=0.95,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask/stream",
            json={"question": "把服务器重启一下"},
        )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[0][0] == "start"
    error_data = next(data for event, data in events if event == "error")
    assert error_data["code"] == "REFUSED"
    assert error_data["message"] == "抱歉，我暂时无法回答这个问题。"
