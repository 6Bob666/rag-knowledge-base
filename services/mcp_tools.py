"""MCP 工具与 Function Calling 工具之间的转换与调用适配。

两套协议描述的是同一件事，但字段名不同：

    MCP tools/list           OpenAI/DeepSeek Function Calling
    ----------------------   ----------------------------------
    name                     function.name
    description              function.description
    inputSchema              function.parameters

有了这一层，Agent 就不再手写工具定义：工具由 MCP Server 声明，
模型按标准 tools 协议调用，代码再把调用转发回 MCP Server 执行。

    MCP tools/list ──转换──▶ FC tools ──▶ LLM 返回 tool_calls
                                              │
                        MCP tools/call ◀──────┘
"""

import threading
from typing import Any, Protocol


class MCPClientLike(Protocol):
    """MCPToolProvider 需要的最小客户端接口，便于测试替换。"""

    def list_tools(self) -> list[dict[str, Any]]: ...

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> str: ...


def mcp_tool_to_function_calling(tool: dict[str, Any]) -> dict[str, Any]:
    """把单个 MCP 工具描述转换成 Function Calling 工具定义。"""
    name = str(tool.get("name") or "").strip()
    if not name:
        raise ValueError("MCP 工具缺少 name")

    schema = tool.get("inputSchema")
    if not isinstance(schema, dict):
        # 没有声明参数的 MCP 工具，对模型表现为"无参数工具"。
        schema = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": str(tool.get("description") or ""),
            "parameters": schema,
        },
    }


def convert_mcp_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """批量转换；跳过非法工具，避免一个坏 schema 让整个 Agent 起不来。"""
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        try:
            converted.append(mcp_tool_to_function_calling(tool))
        except ValueError:
            continue
    return converted


def function_calling_tool_names(tools: list[dict[str, Any]]) -> tuple[str, ...]:
    """从 Function Calling 工具定义中取出工具名白名单。"""
    names = []
    for tool in tools or []:
        function = tool.get("function") or {}
        name = str(function.get("name") or "").strip()
        if name:
            names.append(name)
    return tuple(names)


class MCPToolProvider:
    """把 MCP Client 包装成 Agent 可用的工具源：列工具 + 调工具。"""

    def __init__(self, client: MCPClientLike, *, cache_tools: bool = True):
        self._client = client
        self._cache_tools = cache_tools
        self._tools: list[dict[str, Any]] | None = None
        self._lock = threading.Lock()

    def function_calling_tools(self) -> list[dict[str, Any]]:
        """返回 Function Calling 格式的工具列表（默认缓存一次）。"""
        with self._lock:
            if self._tools is None or not self._cache_tools:
                self._tools = convert_mcp_tools(self._client.list_tools())
            return list(self._tools)

    def allowed_tool_names(self) -> tuple[str, ...]:
        return function_calling_tool_names(self.function_calling_tools())

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return self._client.call_tool(name, arguments)
