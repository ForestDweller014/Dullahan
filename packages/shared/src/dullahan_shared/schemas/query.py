from __future__ import annotations

from pydantic import BaseModel, Field

from dullahan_shared.schemas.context import ContextBundle


class QueryEnvelope(BaseModel):
    sender_id: str
    query_id: str
    query: str = Field(min_length=1)
    parent_context: ContextBundle | None = None
    depth: int = Field(default=0, ge=0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class SubqueryPlan(BaseModel):
    parent_query_id: str
    subqueries: list[QueryEnvelope] = Field(default_factory=list)
    rationale_summary: str | None = None
