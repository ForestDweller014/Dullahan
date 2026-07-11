from __future__ import annotations

from pydantic import BaseModel, Field

from dullahan_shared.schemas.context import ContextBundle
from dullahan_shared.schemas.expert import ExpertResponse


class DispatchRequest(BaseModel):
    sender_id: str
    query_id: str
    subquery: str = Field(min_length=1)
    context: ContextBundle


class DispatchResponse(BaseModel):
    response: ExpertResponse


class BatchDispatchRequest(BaseModel):
    requests: list[DispatchRequest] = Field(min_length=1)


class BatchDispatchResponse(BaseModel):
    responses: list[ExpertResponse]
