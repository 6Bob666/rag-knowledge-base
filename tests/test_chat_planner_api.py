"""Planner / Verifier 接入 /chat/ask 后的接口级行为测试。"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.chat import (
    get_llm,
    get_planner,
    get_query_rewriter,
    get_reranker,
    get_verifier,
    router,
)
from services.conversation_store import InMemoryConversationStore
from services.dependencies import get_conversation_store, get_store
from services.plan_verify import (
    ACTION_GENERATE,
    ACTION_RETRIEVE,
    AlwaysSufficientVerifier,
    PlanStep,
)


ON_TOPIC_TEXT = "CNN 的全称是 Convolutional Neural Network。"
OFF_TOPIC_TEXT = "今天天气不错，适合出门散步。"


def record(identifier: str, text: str) -> dict:
    return {
        "id": identifier,
        "text": text,
        "metadata": {"source": "ai.txt", "chunk_id": identifier},
    }


class OffTopicThenHitStore:
    """只有"关键词查询"能命中，用来区分两种补救策略。"""

    def __init__(self):
        self.calls: list[str] = []

    def hybrid_search_records(self, query, top_k):
        self.calls.append(query)
        if query == "cnn 全称":
            return [record("hit", ON_TOPIC_TEXT)]
        return [record("off", OFF_TOPIC_TEXT)]


class AlwaysEmptyStore:
    def __init__(self):
        self.calls: list[str] = []

    def hybrid_search_records(self, query, top_k):
        self.calls.append(query)
        return []


class FakeReranker:
    def rerank_records(self, query, records, top_k):
        return records[:top_k]


def build_app(store, *, verifier=None, planner=None, rewriter=None):
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_query_rewriter] = lambda: (
        rewriter or (lambda question: question)
    )
    app.dependency_overrides[get_llm] = lambda: (
        lambda question, context: "这是基于检索结果生成的回答。"
    )
    app.dependency_overrides[get_conversation_store] = lambda: (
        InMemoryConversationStore()
    )
    if verifier is not None:
        app.dependency_overrides[get_verifier] = lambda: verifier
    if planner is not None:
        app.dependency_overrides[get_planner] = lambda: planner
    return app


def test_off_topic_result_triggers_keyword_replan():
    """首检跑题时，第二次检索换关键词查询，而不是重复原问题。"""
    store = OffTopicThenHitStore()
    app = build_app(
        store,
        rewriter=lambda question: f"改写后：{question}",
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "CNN 的全称是什么？", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "这是基于检索结果生成的回答。"
    # 旧逻辑第二轮会搜原问题（同样跑题→拒答），现在改搜关键词查询并命中。
    assert store.calls == ["改写后：CNN 的全称是什么？", "cnn 全称"]
    assert response.json()["contexts"][0]["chunk_id"] == "hit"


def test_disabling_planner_restores_single_pass_behaviour():
    """回归开关：退回旧行为时，有上下文就直接生成，不再看覆盖率。"""
    store = OffTopicThenHitStore()
    app = build_app(store, verifier=AlwaysSufficientVerifier())

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "CNN 的全称是什么？", "top_k": 3},
        )

    assert response.status_code == 200
    assert store.calls == ["CNN 的全称是什么？"]
    assert response.json()["contexts"][0]["chunk_id"] == "off"


def test_retry_is_bounded_and_refuses_when_nothing_found():
    store = AlwaysEmptyStore()
    app = build_app(store, rewriter=lambda question: f"改写后：{question}")

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "完全不在知识库的问题", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "未找到相关文档。"
    # 只有两次检索预算，搜不到就停，不会因为策略池里还有别的问法而继续搜。
    assert len(store.calls) == 2


def test_buggy_planner_cannot_break_attempt_limit():
    """即使 Planner 被换成只会说"继续检索"的实现，也不能无限循环。"""

    class GreedyPlanner:
        def plan(self, *, state, verification, strategies):
            return PlanStep(ACTION_RETRIEVE, query="再搜一次", reason="greedy")

    store = AlwaysEmptyStore()
    app = build_app(store, planner=GreedyPlanner())

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "CNN 的全称是什么？", "top_k": 3},
        )

    assert response.status_code == 200
    assert len(store.calls) == 2


def test_planner_records_verification_trace_in_logs(caplog):
    import logging

    store = OffTopicThenHitStore()
    app = build_app(store, rewriter=lambda question: f"改写后：{question}")

    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            client.post(
                "/chat/ask",
                json={"question": "CNN 的全称是什么？", "top_k": 3},
            )

    trace_lines = [item for item in caplog.text.splitlines() if "rag_answer" in item]
    assert trace_lines
    # 计划与验证结论要能被日志聚合工具解析出来：
    # 第一次跑题、第二次命中，整条轨迹都要留在日志里。
    assert "verify_trail=low_coverage>sufficient" in caplog.text
    assert "plan_trail=retrieve>generate" in caplog.text
    assert "plan_action=generate" in caplog.text


def test_sufficient_first_attempt_skips_second_search():
    class OnTopicStore:
        def __init__(self):
            self.calls: list[str] = []

        def hybrid_search_records(self, query, top_k):
            self.calls.append(query)
            return [record("hit", ON_TOPIC_TEXT)]

    store = OnTopicStore()
    app = build_app(store)

    with TestClient(app) as client:
        response = client.post(
            "/chat/ask",
            json={"question": "CNN 的全称是什么？", "top_k": 3},
        )

    assert response.status_code == 200
    assert store.calls == ["CNN 的全称是什么？"]
