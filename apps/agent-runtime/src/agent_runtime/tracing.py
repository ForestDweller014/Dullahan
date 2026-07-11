from __future__ import annotations

from datetime import UTC, datetime
from threading import RLock
from time import perf_counter

from dullahan_shared.ids import new_id
from dullahan_shared.schemas.execution import ExecutionSpan, ExecutionStatus


class InMemoryTraceCollector:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or new_id("trace")
        self.spans: list[ExecutionSpan] = []
        self._starts: dict[str, float] = {}
        self._lock = RLock()

    def start_span(
        self,
        *,
        name: str,
        query_id: str | None = None,
        parent_span_id: str | None = None,
        parent_query_id: str | None = None,
        depth: int = 0,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> ExecutionSpan:
        span = ExecutionSpan(
            trace_id=self.trace_id,
            span_id=new_id("span"),
            name=name,
            query_id=query_id,
            parent_span_id=parent_span_id,
            parent_query_id=parent_query_id,
            depth=depth,
            status=ExecutionStatus.RUNNING,
            attributes=attributes or {},
        )
        with self._lock:
            self.spans.append(span)
            self._starts[span.span_id] = perf_counter()
        return span

    def end_span(
        self,
        span: ExecutionSpan,
        *,
        status: ExecutionStatus = ExecutionStatus.SUCCEEDED,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> ExecutionSpan:
        with self._lock:
            elapsed_ms = (perf_counter() - self._starts.pop(span.span_id, perf_counter())) * 1000
            updated_attributes = {**span.attributes, **(attributes or {})}
            updated = span.model_copy(
                update={
                    "status": status,
                    "ended_at": datetime.now(UTC),
                    "duration_ms": round(elapsed_ms, 6),
                    "attributes": updated_attributes,
                }
            )
            index = self.spans.index(span)
            self.spans[index] = updated
            return updated
