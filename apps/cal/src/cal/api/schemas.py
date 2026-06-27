from __future__ import annotations

from pydantic import BaseModel, Field

from dullahan_shared.schemas.context import ContextBundle


class AugmentContextRequest(BaseModel):
    sender_id: str
    query_id: str | None = None
    subquery: str = Field(min_length=1)
    parent_context: ContextBundle


class AugmentContextResponse(BaseModel):
    subquery: str
    context: ContextBundle


class BatchAugmentContextRequest(BaseModel):
    requests: list[AugmentContextRequest] = Field(min_length=1)


class BatchAugmentContextResponse(BaseModel):
    responses: list[AugmentContextResponse]
