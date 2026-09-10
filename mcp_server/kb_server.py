"""最小 MCP Server 实现：把知识库检索暴露为 MCP 工具。

不依赖官方 mcp SDK，直接实现 MCP 使用的 JSON-RPC 2.0 + stdio 协议，
这样本地开发、测试和面试演示都能直接运行：

    python -m mcp_server.kb_server

支持的方法：
    initialize / ping / tools/list / tools/call
"""

import json
import sys
from typing import Any, Callable


MCP_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "rag-knowledge-base-mcp"
SERVER_VERSION = "0.1.0"


class MCPError(Exception):
    """JSON-RPC 协议级错误。"""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ToolRegistry:
    """工具注册表：工具名到处理函数的显式白名单。"""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[[dict[str, Any]], str],
    ) -> None:
        if not name:
            raise ValueError("工具名不能为空")
        self._tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "handler": handler,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise MCPError(-32602, f"未知工具: {name}")
        return tool["handler"](arguments or {})


def _success(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


def handle_message(
    message: dict[str, Any],
    registry: ToolRegistry,
) -> dict[str, Any] | None:
    """处理一条 JSON-RPC 消息；通知类消息返回 None。"""
    if not isinstance(message, dict):
        return _error(None, -32600, "请求必须是 JSON 对象")

    method = message.get("method")
    message_id = message.get("id")

    # 没有 id 的是 notification，例如 notifications/initialized，不需要响应。
    if message_id is None:
        return None

    if method == "initialize":
        requested = (message.get("params") or {}).get("protocolVersion")
        return _success(
            message_id,
            {
                "protocolVersion": requested or MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )

    if method == "ping":
        return _success(message_id, {})

    if method == "tools/list":
        return _success(message_id, {"tools": registry.list_tools()})

    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name") or ""
        arguments = params.get("arguments") or {}
        try:
            content = registry.call(name, arguments)
        except MCPError as exc:
            return _success(
                message_id,
                {
                    "content": [{"type": "text", "text": exc.message}],
                    "isError": True,
                },
            )
        except Exception as exc:  # 工具内部异常不暴露堆栈
            return _success(
                message_id,
                {
                    "content": [
                        {"type": "text", "text": f"工具执行失败: {exc}"}
                    ],
                    "isError": True,
                },
            )
        return _success(
            message_id,
            {"content": [{"type": "text", "text": content}], "isError": False},
        )

    return _error(message_id, -32601, f"不支持的方法: {method}")


def build_default_registry() -> ToolRegistry:
    """构建使用真实知识库的默认工具注册表。"""
    registry = ToolRegistry()
    services: dict[str, Any] = {}

    def get_services():
        if "store" not in services:
            from services.vector_store import VectorStore

            services["store"] = VectorStore()
        if "reranker" not in services:
            from services.reranker import ReRanker

            services["reranker"] = ReRanker()
        return services["store"], services["reranker"]

    def search_knowledge_base(arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise MCPError(-32602, "query 不能为空")
        top_k = int(arguments.get("top_k", 3))
        if not 1 <= top_k <= 10:
            raise MCPError(-32602, "top_k 必须在 1 到 10 之间")

        store, reranker = get_services()
        candidates = store.hybrid_search_records(query, top_k=top_k * 3)
        contexts = reranker.rerank_records(query, candidates, top_k)
        return json.dumps(
            [
                {
                    "text": item.get("text", ""),
                    "source": (item.get("metadata") or {}).get(
                        "source", "unknown"
                    ),
                    "chunk_id": str(
                        (item.get("metadata") or {}).get(
                            "chunk_id", item.get("id", "unknown")
                        )
                    ),
                }
                for item in contexts
            ],
            ensure_ascii=False,
        )

    def list_knowledge_documents(arguments: dict[str, Any]) -> str:
        store, _ = get_services()
        return json.dumps(store.list_documents(), ensure_ascii=False)

    registry.register(
        name="search_knowledge_base",
        description="在本地知识库中检索与问题相关的文档片段。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问法"},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                },
            },
            "required": ["query"],
        },
        handler=search_knowledge_base,
    )
    registry.register(
        name="list_knowledge_documents",
        description="列出当前知识库中已入库的文档。",
        input_schema={"type": "object", "properties": {}},
        handler=list_knowledge_documents,
    )
    return registry


def run_stdio(registry: ToolRegistry | None = None) -> None:
    """按行读取 JSON-RPC 消息并输出响应，供 MCP Client 通过 stdio 调用。"""
    registry = registry or build_default_registry()
    # 显式使用 readline，避免 for 迭代在管道上的预读缓冲导致消息被卡住。
    for stream in (sys.stdin, sys.stdout):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    while True:
        raw_line = sys.stdin.readline()
        if raw_line == "":
            break
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            # 无法解析时没有可用的请求 id，只能忽略这一行。
            continue
        response = handle_message(message, registry)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def main() -> None:
    run_stdio()


if __name__ == "__main__":
    main()
