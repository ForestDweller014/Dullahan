from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TextIO


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def as_mcp_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class JsonRpcMcpServer:
    def __init__(
        self,
        *,
        name: str,
        tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
        tool_specs: Iterable[ToolSpec],
        protocol_version: str = "2024-11-05",
    ) -> None:
        self.name = name
        self.tools = tools
        self.tool_specs = {spec.name: spec for spec in tool_specs}
        self.protocol_version = protocol_version

    def serve(self, *, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        for line in stdin:
            if not line.strip():
                continue
            response = self.handle_json(line)
            if response is None:
                continue
            stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            stdout.flush()

    def handle_json(self, line: str) -> dict[str, Any] | None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            return self._error(None, -32700, f"parse error: {exc.msg}")
        return self.handle_message(message)

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        message_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}

        if message_id is None:
            return None
        if method == "initialize":
            return self._result(
                message_id,
                {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": "0.1.0"},
                },
            )
        if method == "tools/list":
            return self._result(
                message_id,
                {"tools": [spec.as_mcp_tool() for spec in self.tool_specs.values()]},
            )
        if method == "tools/call":
            return self._call_tool(message_id, params)
        return self._error(message_id, -32601, f"unknown method: {method}")

    def _call_tool(self, message_id: str | int, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        tool = self.tools.get(name)
        if tool is None:
            return self._error(message_id, -32602, f"unknown tool: {name}")
        try:
            result = tool(arguments)
        except Exception as exc:
            return self._result(
                message_id,
                {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
            )
        return self._result(
            message_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, sort_keys=True),
                    }
                ],
                "structuredContent": result,
                "isError": False,
            },
        )

    def _result(self, message_id: str | int, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    def _error(
        self,
        message_id: str | int | None,
        code: int,
        message: str,
    ) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "error": {"code": code, "message": message},
        }
