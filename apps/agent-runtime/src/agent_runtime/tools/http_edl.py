from __future__ import annotations

from dullahan_shared.schemas.context import ContextBundle
from dullahan_shared.schemas.expert import ExpertResponse
from dullahan_shared.schemas.query import QueryEnvelope
from edl.api.schemas import (
    BatchDispatchRequest,
    BatchDispatchResponse,
    DispatchRequest,
    DispatchResponse,
)

from agent_runtime.tools.http import post_json


class HttpEdlTool:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def send(self, subquery: QueryEnvelope, context: ContextBundle) -> ExpertResponse:
        return post_json(
            url=f"{self.base_url}/dispatch",
            payload=DispatchRequest(
                sender_id=subquery.sender_id,
                query_id=subquery.query_id,
                subquery=subquery.query,
                context=context,
            ),
            response_model=DispatchResponse,
            timeout_seconds=self.timeout_seconds,
        ).response

    def send_batch(
        self,
        items: list[tuple[QueryEnvelope, ContextBundle]],
    ) -> list[ExpertResponse]:
        return post_json(
            url=f"{self.base_url}/dispatch/batch",
            payload=BatchDispatchRequest(
                requests=[
                    DispatchRequest(
                        sender_id=subquery.sender_id,
                        query_id=subquery.query_id,
                        subquery=subquery.query,
                        context=context,
                    )
                    for subquery, context in items
                ]
            ),
            response_model=BatchDispatchResponse,
            timeout_seconds=self.timeout_seconds,
        ).responses
