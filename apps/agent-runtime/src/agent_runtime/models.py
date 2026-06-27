from __future__ import annotations

from pydantic import BaseModel, Field

from dullahan_shared.schemas.context import ContextBundle
from dullahan_shared.schemas.execution import ExecutionSpan
from dullahan_shared.schemas.expert import ExpertResponse
from dullahan_shared.schemas.query import QueryEnvelope


class AgentRunRequest(BaseModel):
    sender_id: str = "user"
    query: str = Field(min_length=1)
    persist_artifacts: bool = False


class AgentRunResult(BaseModel):
    root_query: QueryEnvelope
    subqueries: list[QueryEnvelope]
    contexts: list[ContextBundle] = Field(default_factory=list)
    expert_responses: list[ExpertResponse]
    trace_id: str
    spans: list[ExecutionSpan]
    final_response: str
    artifact_dir: str | None = None
