from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.chat import get_reranker, get_tool_agent_responder, router
from services.conversation_store import InMemoryConversationStore
from services.dependencies import get_conversation_store, get_store


class FakeStore:
    def __init__(self):
        self.queries = []

    def hybrid_search_records(self, query, top_k):
        self.queries.append(query)
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


def build_app(responder, store=None):
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    app.dependency_overrides[get_store] = lambda: store or FakeStore()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_tool_agent_responder] = lambda: responder
    app.dependency_overrides[get_conversation_store] = lambda: (
        InMemoryConversationStore()
    )
    return app


def tool_call(call_id, query):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "arguments": f'{{"query": "{query}"}}',
        },
    }


def test_tool_agent_endpoint_searches_then_answers():
    state = {"round": 0}
    store = FakeStore()

    def fake_responder(messages):
        state["round"] += 1
        if state["round"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    tool_call("call_1", "CNN 的全称是什么？")
                ],
            }
        return {
            "content": "CNN 的全称是卷积神经网络。",
            "tool_calls": [],
        }

    app = build_app(fake_responder, store=store)

    with TestClient(app) as client:
        response = client.post(
            "/chat/agent/tool",
            json={"question": "CNN 的全称是什么？", "top_k": 1},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "CNN 的全称是卷积神经网络。"
    assert store.queries == ["CNN 的全称是什么？"]
    assert response.json()["contexts"] == [
        {
            "text": "CNN 的全称是 Convolutional Neural Network。",
            "source": "ai.txt",
            "chunk_id": "chunk-1",
        }
    ]
    assert response.headers["X-Agent-Tool-Calls"] == "1"
    assert response.headers["X-Agent-Termination"] == "answered"


def test_tool_agent_endpoint_can_answer_without_tool():
    def fake_responder(messages):
        return {"content": "你好！", "tool_calls": []}

    class ExplodingStore:
        def hybrid_search_records(self, query, top_k):
            raise AssertionError("未声明工具调用时不应检索")

    app = build_app(fake_responder, store=ExplodingStore())

    with TestClient(app) as client:
        response = client.post(
            "/chat/agent/tool",
            json={"question": "你好"},
        )

    assert response.status_code == 200
    assert response.json() == {"answer": "你好！", "contexts": []}


def test_tool_agent_endpoint_returns_no_answer_at_max_steps():
    def fake_responder(messages):
        return {
            "content": "",
            "tool_calls": [
                tool_call("call_loop", "继续检索")
            ],
        }

    app = build_app(fake_responder)

    with TestClient(app) as client:
        response = client.post(
            "/chat/agent/tool",
            json={"question": "循环问题", "top_k": 1},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "未能在限定步数内给出可靠答案。"
    assert response.json()["contexts"] == []
    assert response.headers["X-Agent-Termination"] == "max_steps"


def test_tool_agent_endpoint_returns_503_when_llm_fails():
    def failing_responder(messages):
        raise RuntimeError("模拟 LLM 故障")

    app = build_app(failing_responder)

    with TestClient(app) as client:
        response = client.post(
            "/chat/agent/tool",
            json={"question": "CNN 是什么？"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "问答模型暂时不可用"
