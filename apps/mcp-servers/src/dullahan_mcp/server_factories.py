from __future__ import annotations

import os

from agent_runtime.tools.http_cal import HttpCalTool
from agent_runtime.tools.http_edl import HttpEdlTool
from dullahan_mcp.handlers import CalMcpHandler, EdlMcpHandler
from dullahan_mcp.stdio import JsonRpcMcpServer, ToolSpec


SEND_TO_CAL_SPEC = ToolSpec(
    name="send_to_CAL",
    description="Augment a subquery with parent-context and WorldStateDB retrieval.",
    input_schema={
        "type": "object",
        "required": ["sender_id", "subquery", "parent_context"],
        "properties": {
            "sender_id": {"type": "string"},
            "query_id": {"type": "string"},
            "subquery": {"type": "string", "minLength": 1},
            "parent_context": {"type": "object"},
        },
    },
)

SEND_TO_EDL_SPEC = ToolSpec(
    name="send_to_EDL",
    description="Dispatch a contextualized subquery to the highest-attention expert.",
    input_schema={
        "type": "object",
        "required": ["sender_id", "query_id", "subquery", "context"],
        "properties": {
            "sender_id": {"type": "string"},
            "query_id": {"type": "string"},
            "subquery": {"type": "string", "minLength": 1},
            "context": {"type": "object"},
        },
    },
)


def build_cal_handler() -> CalMcpHandler:
    return CalMcpHandler(HttpCalTool(os.getenv("CAL_BASE_URL", "http://127.0.0.1:8100")))


def build_edl_handler() -> EdlMcpHandler:
    return EdlMcpHandler(HttpEdlTool(os.getenv("EDL_BASE_URL", "http://127.0.0.1:8200")))


def build_cal_stdio_server() -> JsonRpcMcpServer:
    handler = build_cal_handler()
    return JsonRpcMcpServer(
        name="dullahan-cal",
        tools=handler.tools,
        tool_specs=[SEND_TO_CAL_SPEC],
    )


def build_edl_stdio_server() -> JsonRpcMcpServer:
    handler = build_edl_handler()
    return JsonRpcMcpServer(
        name="dullahan-edl",
        tools=handler.tools,
        tool_specs=[SEND_TO_EDL_SPEC],
    )


def cal_stdio_main() -> None:
    build_cal_stdio_server().serve()


def edl_stdio_main() -> None:
    build_edl_stdio_server().serve()
