from __future__ import annotations

from dullahan_shared.schemas.context import ContextBundle
from dullahan_shared.schemas.expert import ExpertResponse
from dullahan_shared.schemas.query import QueryEnvelope
from edl.api.schemas import DispatchRequest
from edl.service import ExpertDispatchService


class LocalEdlTool:
    def __init__(self, service: ExpertDispatchService) -> None:
        self.service = service

    def send(self, subquery: QueryEnvelope, context: ContextBundle) -> ExpertResponse:
        return self.service.dispatch(
            DispatchRequest(
                sender_id=subquery.sender_id,
                query_id=subquery.query_id,
                subquery=subquery.query,
                context=context,
            )
        ).response
