from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionLimits(BaseModel):
    max_depth: int = Field(default=4, ge=0)
    max_breadth_per_agent: int = Field(default=6, ge=1)
    max_total_agent_instances: int = Field(default=128, ge=1)
    timeout_seconds_per_instance: int = Field(default=60, ge=1)


class ExecutionTrace(BaseModel):
    trace_id: str
    root_query_id: str
    parent_query_id: str | None = None
    depth: int = Field(default=0, ge=0)
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExecutionSpan(BaseModel):
    trace_id: str
    span_id: str
    name: str
    query_id: str | None = None
    parent_span_id: str | None = None
    parent_query_id: str | None = None
    depth: int = Field(default=0, ge=0)
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0.0)
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)
