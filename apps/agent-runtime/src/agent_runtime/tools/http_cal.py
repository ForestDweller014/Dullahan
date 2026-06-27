from __future__ import annotations

from cal.api.schemas import (
    AugmentContextRequest,
    AugmentContextResponse,
    BatchAugmentContextRequest,
    BatchAugmentContextResponse,
)
from dullahan_shared.schemas.context import ContextBundle
from dullahan_shared.schemas.query import QueryEnvelope

from agent_runtime.tools.http import post_json


class HttpCalTool:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def send(self, subquery: QueryEnvelope, parent_context: ContextBundle) -> AugmentContextResponse:
        return post_json(
            url=f"{self.base_url}/augment",
            payload=AugmentContextRequest(
                sender_id=subquery.sender_id,
                query_id=subquery.query_id,
                subquery=subquery.query,
                parent_context=parent_context,
            ),
            response_model=AugmentContextResponse,
            timeout_seconds=self.timeout_seconds,
        )

    def send_batch(
        self,
        items: list[tuple[QueryEnvelope, ContextBundle]],
    ) -> list[AugmentContextResponse]:
        return post_json(
            url=f"{self.base_url}/augment/batch",
            payload=BatchAugmentContextRequest(
                requests=[
                    AugmentContextRequest(
                        sender_id=subquery.sender_id,
                        query_id=subquery.query_id,
                        subquery=subquery.query,
                        parent_context=parent_context,
                    )
                    for subquery, parent_context in items
                ]
            ),
            response_model=BatchAugmentContextResponse,
            timeout_seconds=self.timeout_seconds,
        ).responses
