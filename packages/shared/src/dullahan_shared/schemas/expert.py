from __future__ import annotations

from pydantic import BaseModel, Field


class ExpertProfile(BaseModel):
    id: str
    cluster_id: str
    role_context: str = Field(min_length=1)
    model: str
    max_concurrency: int = Field(default=1, ge=1)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ExpertResponse(BaseModel):
    sender_id: str
    query_id: str
    subquery: str = Field(min_length=1)
    expert_id: str
    response: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    cited_context_document_ids: list[str] = Field(default_factory=list)
    routing_metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
