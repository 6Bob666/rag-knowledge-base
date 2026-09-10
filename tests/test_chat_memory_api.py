from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.chat import get_llm, get_query_rewriter, get_reranker, router
from services.conversation_store import InMemoryConversationStore
from services.dependencies import (
    get_conversation_store,
    get_memory_service,
    get_store,
)
from services.memory_service import MemoryService
from services.memory_store import InMemoryMemoryStore


class FakeStore:
    def hybrid_search_records(self, query, top_k):
        return [
            {
                "id": "chunk-1",
                "text": "RAG 是检索增强生成。",
                "metadata": {"source": "ai.txt", "chunk_id": "chunk-1"},
            }
        ]


class FakeReranker:
    def rerank_records(self, query, records, top_k):
        return records[:top_k]


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def __call__(self, question, context, history=None, memory=""):
        self.calls.append(
            {
                "question": question,
                "context": context,
                "history": history,
                "memory": memory,
            }
        )
        return "这是测试回答"


def build_app(memory_service, llm):
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    app.dependency_overrides[get_store] = lambda: FakeStore()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question, history=None: question
    )
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_conversation_store] = lambda: (
        InMemoryConversationStore()
    )
    app.dependency_overrides[get_memory_service] = lambda: memory_service
    return app


def test_memory_is_remembered_then_recalled_across_turns():
    memory_service = MemoryService(InMemoryMemoryStore(), recall_top_k=3)
    llm = RecordingLLM()
    app = build_app(memory_service, llm)

    with TestClient(app) as client:
        first = client.post(
            "/chat/ask",
            json={
                "question": "我叫刘永超，我研究 RAG。",
                "conversation_id": "memory-001",
                "user_id": "user-001",
            },
        )
        second = client.post(
            "/chat/ask",
            json={
                "question": "RAG 应该怎么学习？",
                "conversation_id": "memory-002",
                "user_id": "user-001",
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    memories = memory_service.store.list("user-001")
    assert any("刘永超" in record.content for record in memories)
    assert "../.." not in memories[0].content

    # 第二轮生成时应该收到长期记忆。
    second_call = llm.calls[-1]
    assert "<user_memory>" in second_call["memory"]
    assert "刘永超" in second_call["memory"] or "RAG" in second_call["memory"]


def test_memory_is_not_written_without_user_id():
    memory_service = MemoryService(InMemoryMemoryStore())
    llm = RecordingLLM()
    app = build_app(memory_service, llm)

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "我叫刘永超，我研究 RAG。"},
        )

    assert response.status_code == 200
    assert memory_service.store.list("user-001") == []
