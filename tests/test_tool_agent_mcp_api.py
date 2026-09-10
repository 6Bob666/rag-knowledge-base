"""/chat/agent/tool 走 MCP 工具通路与降级回本地工具的接口测试。"""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.chat import get_reranker, get_tool_agent_responder, router
from services.conversation_store import InMemoryConversationStore
from services.dependencies import (
    get_conversation_store,
    get_mcp_tool_provider,
    get_store,
)


SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": "在本地知识库中检索与问题相关的文档片段。",
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

MCP_CONTEXTS = [
    {
        "text": "CNN 的全称是 Convolutional Neural Network。",
        "source": "mcp.txt",
        "chunk_id": "mcp-1",
    }
]


class FakeStore:
    """本地 Store；MCP 通路命中时应完全不被调用。"""

    def __init__(self):
        self.queries: list[str] = []

    def hybrid_search_records(self, query, top_k):
        self.queries.append(query)
        return [
            {
                "id": "local-1",
                "text": "本地检索结果",
                "metadata": {"source": "local.txt", "chunk_id": "local-1"},
            }
        ]


class FakeReranker:
    def rerank_records(self, query, records, top_k):
        return records[:top_k]


class FakeMCPProvider:
    def __init__(self, tools=None, result=None, error=None):
        self._tools = tools if tools is not None else [SEARCH_TOOL]
        self._result = result if result is not None else MCP_CONTEXTS
        self._error = error
        self.calls: list[tuple[str, dict]] = []

    def function_calling_tools(self):
        from services.mcp_tools import convert_mcp_tools

        return convert_mcp_tools(self._tools)

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self._error:
            raise self._error
        return json.dumps(self._result, ensure_ascii=False)


def build_app(responder, provider, store):
    app = FastAPI()
    app.include_router(router, prefix="/chat")
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_tool_agent_responder] = lambda: responder
    app.dependency_overrides[get_conversation_store] = lambda: (
        InMemoryConversationStore()
    )
    app.dependency_overrides[get_mcp_tool_provider] = lambda: provider
    return app


def search_call(query, call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "arguments": json.dumps({"query": query}, ensure_ascii=False),
        },
    }


def two_round_responder(query):
    state = {"round": 0}

    def responder(messages, tools):
        state["round"] += 1
        if state["round"] == 1:
            return {"content": "", "tool_calls": [search_call(query)]}
        return {
            "content": "CNN 的全称是 Convolutional Neural Network。",
            "tool_calls": [],
        }

    return responder


def test_agent_tool_endpoint_executes_search_through_mcp():
    provider = FakeMCPProvider()
    store = FakeStore()
    app = build_app(two_round_responder("CNN 全称"), provider, store)

    with TestClient(app) as client:
        response = client.post(
            "/chat/agent/tool",
            json={"question": "CNN 的全称是什么？", "top_k": 3},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("CNN 的全称是")
    # MCP 工具返回的 chunk 被回填进 contexts，而不是本地检索结果。
    assert [item["chunk_id"] for item in body["contexts"]] == ["mcp-1"]
    assert body["contexts"][0]["source"] == "mcp.txt"
    assert store.queries == []
    assert provider.calls == [("search_knowledge_base", {"query": "CNN 全称"})]
    assert response.headers["X-Agent-Tool-Calls"] == "1"


def test_tool_definitions_sent_to_llm_are_converted_from_mcp():
    received: list[list[dict]] = []

    def responder(messages, tools):
        received.append(tools)
        return {"content": "不需要检索", "tool_calls": []}

    app = build_app(responder, FakeMCPProvider(), FakeStore())
    with TestClient(app) as client:
        response = client.post(
            "/chat/agent/tool", json={"question": "你好"}
        )

    assert response.status_code == 200
    assert received[0][0]["function"]["name"] == "search_knowledge_base"
    assert received[0][0]["function"]["parameters"] == SEARCH_TOOL["inputSchema"]


def test_mcp_call_failure_is_returned_to_llm_as_tool_error():
    """MCP 子进程报错不能变成 500，而应以工具错误回填给模型继续推理。"""
    error = RuntimeError("MCP 请求超时: tools/call")
    provider = FakeMCPProvider(error=error)
    tool_payloads: list[str] = []
    state = {"round": 0}

    def responder(messages, tools):
        state["round"] += 1
        tool_payloads.extend(
            message["content"]
            for message in messages
            if message["role"] == "tool"
        )
        if state["round"] == 1:
            return {"content": "", "tool_calls": [search_call("CNN")]}
        return {"content": "知识库暂时不可用。", "tool_calls": []}

    app = build_app(responder, provider, FakeStore())
    with TestClient(app) as client:
        response = client.post(
            "/chat/agent/tool", json={"question": "CNN 的全称是什么？"}
        )

    assert response.status_code == 200
    assert "执行失败" in tool_payloads[0]
    assert response.json()["answer"] == "知识库暂时不可用。"


def test_empty_mcp_tool_list_falls_back_to_local_tool():
    """MCP 没给出任何工具时（关闭或异常），仍走内置本地检索。"""
    provider = FakeMCPProvider(tools=[])
    store = FakeStore()
    app = build_app(two_round_responder("CNN 全称"), provider, store)

    with TestClient(app) as client:
        response = client.post(
            "/chat/agent/tool",
            json={"question": "CNN 的全称是什么？", "top_k": 3},
        )

    assert response.status_code == 200
    assert store.queries == ["CNN 全称"]
    assert [item["chunk_id"] for item in response.json()["contexts"]] == [
        "local-1"
    ]
    assert provider.calls == []


def test_mcp_list_other_than_search_output_is_not_added_to_contexts():
    """list_knowledge_documents 的输出不是检索结果，不应污染 contexts。"""
    provider = FakeMCPProvider(
        result=[{"name": "a.txt", "status": "indexed"}]
    )
    app = build_app(two_round_responder("CNN 全称"), provider, FakeStore())

    with TestClient(app) as client:
        response = client.post(
            "/chat/agent/tool", json={"question": "CNN 的全称是什么？"}
        )

    assert response.status_code == 200
    assert response.json()["contexts"] == []
