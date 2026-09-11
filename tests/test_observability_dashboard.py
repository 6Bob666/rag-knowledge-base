"""可观测性联调：请求 → 日志字段 → Prometheus 指标。"""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.chat import (
    get_llm,
    get_query_rewriter,
    get_reranker,
    router,
)
from routers.metrics import router as metrics_router
from services.conversation_store import InMemoryConversationStore
from services.cost import TokenUsage, record_usage
from services.dependencies import get_conversation_store, get_store
from services.metrics_registry import metrics_registry


class FakeStore:
    def hybrid_search_records(self, query, top_k):
        return [
            {
                "id": "chunk-1",
                "text": "CNN 的全称是 Convolutional Neural Network。",
                "metadata": {"source": "ai.txt", "chunk_id": "chunk-1"},
            }
        ]


class FakeReranker:
    def rerank_records(self, query, records, top_k):
        return records[:top_k]


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics_registry.reset()
    yield
    metrics_registry.reset()


def build_app(store=None):
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    app.include_router(metrics_router)
    app.dependency_overrides[get_store] = lambda: store or FakeStore()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: question
    )
    # 假 LLM 模拟真实 SDK：调用时上报 token 用量。
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_conversation_store] = lambda: (
        InMemoryConversationStore()
    )
    return app


def fake_llm(question, context):
    record_usage(TokenUsage(prompt_tokens=1200, completion_tokens=80))
    return "CNN 的全称是 Convolutional Neural Network。"


def test_trace_line_carries_tokens_and_cost(caplog):
    app = build_app()

    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            response = client.post(
                "/chat/ask", json={"question": "CNN 的全称是什么？"}
            )

    assert response.status_code == 200
    assert "llm_calls=1" in caplog.text
    assert "prompt_tokens=1200" in caplog.text
    assert "completion_tokens=80" in caplog.text
    assert "cost_cny=" in caplog.text


def test_metrics_endpoint_exposes_tokens_and_cost():
    app = build_app()

    with TestClient(app) as client:
        client.post("/chat/ask", json={"question": "CNN 的全称是什么？"})
        body = client.get("/metrics").text

    assert 'rag_llm_tokens_total{type="prompt"} 1200' in body
    assert 'rag_llm_tokens_total{type="completion"} 80' in body
    assert "rag_llm_calls_total 1" in body
    assert "rag_llm_estimated_cost_cny_total" in body
    assert 'rag_planner_verdict_total{verdict="sufficient"} 1' in body
    assert 'rag_planner_action_total{action="generate"} 1' in body


def test_replan_is_counted_in_metrics():
    """第二次检索的请求要能被看板区分出来，并记录首轮失败原因。"""
    calls: list[str] = []

    class OffTopicThenHitStore:
        def hybrid_search_records(self, query, top_k):
            calls.append(query)
            if query == "cnn 全称":
                return [FakeStore().hybrid_search_records(query, top_k)[0]]
            return [
                {
                    "id": "off",
                    "text": "今天天气不错，适合出门散步。",
                    "metadata": {"source": "x.txt", "chunk_id": "off"},
                }
            ]

    app = FastAPI()
    app.include_router(router, prefix="/chat")
    app.include_router(metrics_router)
    app.dependency_overrides[get_store] = lambda: OffTopicThenHitStore()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: f"改写后：{question}"
    )
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_conversation_store] = lambda: (
        InMemoryConversationStore()
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask", json={"question": "CNN 的全称是什么？"}
        )
        body = client.get("/metrics").text

    assert response.status_code == 200
    assert calls == ["改写后：CNN 的全称是什么？", "cnn 全称"]
    assert 'rag_planner_replan_total{first_verdict="low_coverage"} 1' in body


def test_degraded_requests_are_counted():
    class PartiallyCoveredStore:
        def hybrid_search_records(self, query, top_k):
            return [
                {
                    "id": "partial",
                    "text": "CNN 是一种神经网络模型。",
                    "metadata": {"source": "x.txt", "chunk_id": "partial"},
                }
            ]

    app = FastAPI()
    app.include_router(router, prefix="/chat")
    app.include_router(metrics_router)
    app.dependency_overrides[get_store] = lambda: PartiallyCoveredStore()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        lambda question: question
    )
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_conversation_store] = lambda: (
        InMemoryConversationStore()
    )

    with TestClient(app) as client:
        client.post(
            "/chat/ask", json={"question": "CNN 的卷积核是什么", "top_k": 3}
        )
        body = client.get("/metrics").text

    assert "rag_planner_degraded_total 1" in body
