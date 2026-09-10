"""MCP 工具 → Function Calling 工具的转换与适配测试。"""

import pytest

from services.mcp_tools import (
    MCPToolProvider,
    convert_mcp_tools,
    function_calling_tool_names,
    mcp_tool_to_function_calling,
)
from services.tool_agent import run_tool_agent


SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": "在本地知识库中检索与问题相关的文档片段。",
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


def test_mcp_tool_maps_input_schema_to_parameters():
    converted = mcp_tool_to_function_calling(SEARCH_TOOL)

    assert converted["type"] == "function"
    assert converted["function"]["name"] == "search_knowledge_base"
    # MCP 叫 inputSchema，Function Calling 叫 parameters，内容是同一份 schema。
    assert converted["function"]["parameters"] == SEARCH_TOOL["inputSchema"]


def test_tool_without_input_schema_becomes_no_argument_tool():
    converted = mcp_tool_to_function_calling(
        {"name": "list_knowledge_documents"}
    )

    assert converted["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }
    assert converted["function"]["description"] == ""


def test_tool_without_name_is_rejected():
    with pytest.raises(ValueError):
        mcp_tool_to_function_calling({"description": "匿名工具"})


def test_convert_mcp_tools_skips_invalid_entries():
    converted = convert_mcp_tools(
        [SEARCH_TOOL, {"description": "没有名字"}, "not-a-dict", None]
    )

    assert function_calling_tool_names(converted) == ("search_knowledge_base",)


class FakeMCPClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.list_calls = 0

    def list_tools(self):
        self.list_calls += 1
        return [
            SEARCH_TOOL,
            {"name": "list_knowledge_documents", "description": "列出文档"},
        ]

    def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        return '[{"text": "CNN 是全连接网络的误称", "source": "a.txt", ' \
               '"chunk_id": "c1"}]'


def test_provider_caches_tool_list_but_always_calls_client():
    client = FakeMCPClient()
    provider = MCPToolProvider(client)

    first = provider.function_calling_tools()
    second = provider.function_calling_tools()

    # 工具列表是启动期信息，缓存一次即可，不必每次请求都问子进程。
    assert client.list_calls == 1
    assert first == second
    assert provider.allowed_tool_names() == (
        "search_knowledge_base",
        "list_knowledge_documents",
    )

    provider.call_tool("search_knowledge_base", {"query": "CNN"})
    assert client.calls == [("search_knowledge_base", {"query": "CNN"})]


def test_provider_returns_copy_so_caller_cannot_corrupt_cache():
    provider = MCPToolProvider(FakeMCPClient())

    tools = provider.function_calling_tools()
    tools.clear()

    assert provider.function_calling_tools()


def test_run_tool_agent_uses_tools_from_provider_and_whitelists_them():
    """工具白名单由传入的声明推导，而不是写死 search_knowledge_base。"""
    provider = MCPToolProvider(FakeMCPClient())
    tools = provider.function_calling_tools()
    seen_prompts: list[str] = []
    executed: list[tuple[str, dict]] = []

    def responder(messages, received_tools):
        seen_prompts.append(messages[0]["content"])
        assert received_tools == tools
        if not any(message["role"] == "tool" for message in messages):
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "list_knowledge_documents",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        return {"content": "知识库里有 a.txt", "tool_calls": []}

    result = run_tool_agent(
        question="知识库里有哪些文档？",
        execute_tool=lambda name, arguments: executed.append(
            (name, arguments)
        )
        or "[]",
        respond=responder,
        tool_definitions=tools,
    )

    assert result.answer == "知识库里有 a.txt"
    assert result.termination == "answered"
    assert executed == [("list_knowledge_documents", {})]
    assert "list_knowledge_documents" in seen_prompts[0]


def test_run_tool_agent_blocks_tool_outside_declared_whitelist():
    provider = MCPToolProvider(FakeMCPClient())
    tools = provider.function_calling_tools()[:1]  # 只声明检索工具
    executed: list[str] = []
    tool_messages: list[str] = []

    def responder(messages, received_tools):
        tool_messages.extend(
            message["content"]
            for message in messages
            if message["role"] == "tool"
        )
        if not tool_messages:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "list_knowledge_documents",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        return {"content": "好的", "tool_calls": []}

    result = run_tool_agent(
        question="列出文档",
        execute_tool=lambda name, arguments: executed.append(name) or "[]",
        respond=responder,
        tool_definitions=tools,
    )

    assert result.answer == "好的"
    assert executed == []
    assert "不在白名单内" in tool_messages[0]


def test_run_tool_agent_still_supports_single_argument_responder():
    """历史测试替身只接收 messages，签名兼容层要能继续跑通。"""
    rounds = {"count": 0}

    def responder(messages):
        rounds["count"] += 1
        return {"content": "直接回答", "tool_calls": []}

    result = run_tool_agent(
        question="你好",
        execute_tool=lambda name, arguments: "{}",
        respond=responder,
    )

    assert result.answer == "直接回答"
    assert rounds["count"] == 1
