import json

from mcp_server.kb_server import MCPError, ToolRegistry, handle_message


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def echo(arguments):
        return json.dumps({"echo": arguments.get("value", "")}, ensure_ascii=False)

    registry.register(
        name="echo",
        description="回显测试工具",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=echo,
    )
    return registry


def test_initialize_returns_server_info():
    response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        build_registry(),
    )
    assert response["result"]["protocolVersion"] == "2025-06-18"
    assert response["result"]["serverInfo"]["name"] == "rag-knowledge-base-mcp"


def test_tools_list_exposes_schema():
    response = handle_message(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        build_registry(),
    )
    assert response["result"]["tools"][0]["name"] == "echo"
    assert "inputSchema" in response["result"]["tools"][0]


def test_tools_call_returns_text_content():
    response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"value": "hello"}},
        },
        build_registry(),
    )
    assert response["result"]["isError"] is False
    assert response["result"]["content"][0]["text"] == '{"echo": "hello"}'


def test_unknown_tool_is_marked_as_error():
    response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "delete_everything", "arguments": {}},
        },
        build_registry(),
    )
    assert response["result"]["isError"] is True
    assert "未知工具" in response["result"]["content"][0]["text"]


def test_notification_has_no_response():
    assert (
        handle_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            build_registry(),
        )
        is None
    )


def test_unknown_method_returns_jsonrpc_error():
    response = handle_message(
        {"jsonrpc": "2.0", "id": 5, "method": "resources/list"},
        build_registry(),
    )
    assert response["error"]["code"] == -32601
