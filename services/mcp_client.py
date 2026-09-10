"""MCP Client：通过 stdio 连接 MCP Server 并调用工具。"""

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from typing import Any


class MCPClientError(RuntimeError):
    pass


class StdioMCPClient:
    """启动 MCP Server 子进程，按 JSON-RPC 协议发送请求。"""

    def __init__(
        self,
        command: list[str] | None = None,
        *,
        cwd: str | Path | None = None,
        request_timeout: float = 30.0,
        stderr_tail: int = 50,
    ):
        self.command = command or [sys.executable, "-m", "mcp_server.kb_server"]
        self.cwd = Path(cwd) if cwd else Path(__file__).resolve().parents[1]
        self.request_timeout = request_timeout
        self._process: subprocess.Popen | None = None
        self._responses: Queue[dict[str, Any]] = Queue()
        self._stderr_lines: deque[str] = deque(maxlen=stderr_tail)
        self._next_id = 0
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None

    def start(self) -> "StdioMCPClient":
        if self._process is not None:
            return self
        self._process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            },
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        return self

    def _read_stdout(self) -> None:
        assert self._process and self._process.stdout
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def _read_stderr(self) -> None:
        assert self._process and self._process.stderr
        for line in self._process.stderr:
            self._stderr_lines.append(line.rstrip())

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_lines)

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if self._process is None:
            self.start()
        assert self._process and self._process.stdin

        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if params is not None:
                payload["params"] = params
            self._process.stdin.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )
            self._process.stdin.flush()

            deadline = time.monotonic() + (timeout or self.request_timeout)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPClientError(
                        f"MCP 请求超时: {method}; stderr={self.stderr_tail}"
                    )
                try:
                    message = self._responses.get(timeout=remaining)
                except Empty as exc:
                    raise MCPClientError(
                        f"MCP 请求超时: {method}; stderr={self.stderr_tail}"
                    ) from exc
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    error = message["error"]
                    raise MCPClientError(
                        f"MCP 错误 {error.get('code')}: {error.get('message')}"
                    )
                return message.get("result") or {}

    def initialize(self) -> dict[str, Any]:
        return self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "rag-app", "version": "0.1.0"},
            },
        )

    def list_tools(self) -> list[dict[str, Any]]:
        return self.request("tools/list").get("tools", [])

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        result = self.request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )
        texts = [
            item.get("text", "")
            for item in result.get("content", [])
            if item.get("type") == "text"
        ]
        content = "\n".join(texts)
        if result.get("isError"):
            raise MCPClientError(content or "MCP 工具执行失败")
        return content

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._process.stdin and self._process.stdin.close()
        except Exception:
            pass
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None

    def __enter__(self) -> "StdioMCPClient":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
