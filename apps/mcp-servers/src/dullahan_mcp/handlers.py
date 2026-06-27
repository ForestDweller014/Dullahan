from __future__ import annotations

from cal.api.schemas import AugmentContextRequest
from dullahan_shared.schemas.context import ContextBundle
from edl.api.schemas import DispatchRequest

from agent_runtime.tools.http_cal import HttpCalTool
from agent_runtime.tools.http_edl import HttpEdlTool


class CalMcpHandler:
    def __init__(self, tool: HttpCalTool) -> None:
        self.tool = tool

    @property
    def tools(self) -> dict[str, callable]:
        return {"send_to_CAL": self.send_to_cal}

    def send_to_cal(self, arguments: dict) -> dict:
        request = AugmentContextRequest.model_validate(arguments)
        response = self.tool.send(
            subquery=_query_like(
                sender_id=request.sender_id,
                query_id=request.query_id or request.parent_context.query_id,
                query=request.subquery,
            ),
            parent_context=request.parent_context,
        )
        return response.model_dump(mode="json")


class EdlMcpHandler:
    def __init__(self, tool: HttpEdlTool) -> None:
        self.tool = tool

    @property
    def tools(self) -> dict[str, callable]:
        return {"send_to_EDL": self.send_to_edl}

    def send_to_edl(self, arguments: dict) -> dict:
        request = DispatchRequest.model_validate(arguments)
        response = self.tool.send(
            subquery=_query_like(
                sender_id=request.sender_id,
                query_id=request.query_id,
                query=request.subquery,
            ),
            context=request.context,
        )
        return response.model_dump(mode="json")


def _query_like(*, sender_id: str, query_id: str, query: str):
    from dullahan_shared.schemas.query import QueryEnvelope

    return QueryEnvelope(
        sender_id=sender_id,
        query_id=query_id,
        query=query,
        parent_context=ContextBundle(query_id=query_id),
    )
