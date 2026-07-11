from __future__ import annotations

from cal.api.schemas import AugmentContextRequest, AugmentContextResponse
from cal.service import ContextAugmentationService
from dullahan_shared.schemas.context import ContextBundle
from dullahan_shared.schemas.query import QueryEnvelope


class LocalCalTool:
    def __init__(self, service: ContextAugmentationService) -> None:
        self.service = service

    def send(self, subquery: QueryEnvelope, parent_context: ContextBundle) -> AugmentContextResponse:
        return self.service.augment(
            AugmentContextRequest(
                sender_id=subquery.sender_id,
                query_id=subquery.query_id,
                subquery=subquery.query,
                parent_context=parent_context,
            )
        )
