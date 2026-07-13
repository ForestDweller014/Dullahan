from __future__ import annotations

from cal.api.schemas import AugmentContextResponse
from dullahan_mcp.handlers import CalMcpHandler, EdlMcpHandler
from dullahan_mcp.server_factories import SEND_TO_CAL_SPEC, SEND_TO_EDL_SPEC
from dullahan_mcp.stdio import JsonRpcMcpServer
from dullahan_shared.schemas.context import ContextBundle, ContextDocument, ContextSource
from dullahan_shared.schemas.expert import ExpertResponse


class FakeCalTool:
    def __init__(self) -> None:
        self.calls = []

    def send(self, subquery, parent_context):
        self.calls.append({"subquery": subquery, "parent_context": parent_context})
        return AugmentContextResponse(
            subquery=subquery.query,
            context=ContextBundle(
                query_id=subquery.query_id,
                documents=[
                    ContextDocument(
                        id="doc:cal",
                        source=ContextSource.WORLD_STATE,
                        text="CAL context",
                    )
                ],
            ),
        )


class FakeEdlTool:
    def __init__(self) -> None:
        self.calls = []

    def send(self, subquery, context):
        self.calls.append({"subquery": subquery, "context": context})
        return ExpertResponse(
            sender_id=subquery.sender_id,
            query_id=subquery.query_id,
            subquery=subquery.query,
            expert_id="expert:context_memory",
            response="handled",
            confidence=0.9,
        )


# Verifies that send to CAL handler validates and returns JSON shape.
def test_send_to_cal_handler_validates_and_returns_json_shape() -> None:
    tool = FakeCalTool()
    handler = CalMcpHandler(tool)

    result = handler.send_to_cal(
        {
            "sender_id": "query:root",
            "query_id": "query:child",
            "subquery": "How should CAL retrieve context?",
            "parent_context": {
                "query_id": "query:root",
                "documents": [],
                "token_budget": 1024,
            },
        }
    )

    assert result["subquery"] == "How should CAL retrieve context?"
    assert result["context"]["query_id"] == "query:child"
    assert result["context"]["documents"][0]["id"] == "doc:cal"
    assert tool.calls[0]["subquery"].sender_id == "query:root"
    assert tool.calls[0]["subquery"].query_id == "query:child"
    assert "send_to_CAL" in handler.tools


# Verifies that send to EDL handler validates and returns JSON shape.
def test_send_to_edl_handler_validates_and_returns_json_shape() -> None:
    tool = FakeEdlTool()
    handler = EdlMcpHandler(tool)

    result = handler.send_to_edl(
        {
            "sender_id": "query:root",
            "query_id": "query:child",
            "subquery": "Which expert should answer?",
            "context": {
                "query_id": "query:child",
                "documents": [],
            },
        }
    )

    assert result["expert_id"] == "expert:context_memory"
    assert result["response"] == "handled"
    assert tool.calls[0]["context"].query_id == "query:child"
    assert "send_to_EDL" in handler.tools


# Verifies that the stdio MCP server returns its tool definition from tools/list.
def test_stdio_server_lists_tools() -> None:
    handler = CalMcpHandler(FakeCalTool())
    server = JsonRpcMcpServer(
        name="dullahan-cal",
        tools=handler.tools,
        tool_specs=[SEND_TO_CAL_SPEC],
    )

    response = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert response["result"]["tools"][0]["name"] == "send_to_CAL"
    assert response["result"]["tools"][0]["inputSchema"]["required"] == [
        "sender_id",
        "subquery",
        "parent_context",
    ]


# Verifies that stdio server calls CAL tool and returns structured content.
def test_stdio_server_calls_cal_tool_and_returns_structured_content() -> None:
    tool = FakeCalTool()
    handler = CalMcpHandler(tool)
    server = JsonRpcMcpServer(
        name="dullahan-cal",
        tools=handler.tools,
        tool_specs=[SEND_TO_CAL_SPEC],
    )

    response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": "call-1",
            "method": "tools/call",
            "params": {
                "name": "send_to_CAL",
                "arguments": {
                    "sender_id": "query:root",
                    "query_id": "query:child",
                    "subquery": "Retrieve context",
                    "parent_context": {"query_id": "query:root", "documents": []},
                },
            },
        }
    )

    assert response["id"] == "call-1"
    assert response["result"]["isError"] is False
    assert response["result"]["structuredContent"]["context"]["documents"][0]["id"] == "doc:cal"
    assert response["result"]["structuredContent"]["context"]["query_id"] == "query:child"
    assert response["result"]["content"][0]["type"] == "text"
    assert tool.calls[0]["subquery"].query == "Retrieve context"


# Verifies that stdio server calls EDL tool and reports unknown tools.
def test_stdio_server_calls_edl_tool_and_reports_unknown_tools() -> None:
    handler = EdlMcpHandler(FakeEdlTool())
    server = JsonRpcMcpServer(
        name="dullahan-edl",
        tools=handler.tools,
        tool_specs=[SEND_TO_EDL_SPEC],
    )

    response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "send_to_EDL",
                "arguments": {
                    "sender_id": "query:root",
                    "query_id": "query:child",
                    "subquery": "Route this",
                    "context": {"query_id": "query:child", "documents": []},
                },
            },
        }
    )
    unknown_response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "missing", "arguments": {}},
        }
    )

    assert response["result"]["structuredContent"]["expert_id"] == "expert:context_memory"
    assert unknown_response["error"]["code"] == -32602
