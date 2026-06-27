from pathlib import Path
from threading import Lock
from time import perf_counter, sleep

from agent_runtime.aggregation import ResponseAggregator
from agent_runtime.agent import AgentRuntime
from agent_runtime.config import AgentRuntimeConfig
from agent_runtime.models import AgentRunRequest
from agent_runtime.planning.subquery_generator import DeterministicSubqueryGenerator
from agent_runtime.recursion import RecursionGuard
from cal.api.schemas import AugmentContextResponse
from dullahan_shared.schemas.context import ContextBundle
from dullahan_shared.schemas.execution import ExecutionLimits, ExecutionStatus
from dullahan_shared.schemas.expert import ExpertResponse
from dullahan_shared.schemas.query import QueryEnvelope


ROOT = Path(__file__).resolve().parents[3]


def test_agent_runtime_calls_cal_and_edl_for_subqueries() -> None:
    runtime = AgentRuntime.local(
        AgentRuntimeConfig(
            repo_root=ROOT,
            limits=ExecutionLimits(max_depth=2, max_breadth_per_agent=2),
        )
    )

    result = runtime.run(
        AgentRunRequest(query="How should hierarchical execution use CAL and EDL?")
    )

    assert len(result.subqueries) == 6
    assert len(result.expert_responses) == 6
    assert "Root query:" in result.final_response
    assert {subquery.depth for subquery in result.subqueries} == {1, 2}
    assert any(response.sender_id == result.root_query.query_id for response in result.expert_responses)
    assert result.trace_id.startswith("trace:")
    assert result.subqueries[0].metadata["generated_by"] == "deterministic-planner"
    assert {span.name for span in result.spans} >= {
        "agent.run",
        "agent.subquery",
        "cal.augment",
        "edl.dispatch",
    }


def test_agent_runtime_honors_depth_limit() -> None:
    runtime = AgentRuntime.local(
        AgentRuntimeConfig(
            repo_root=ROOT,
            limits=ExecutionLimits(max_depth=0, max_breadth_per_agent=3),
        )
    )

    result = runtime.run(AgentRunRequest(query="Will this recurse?"))

    assert result.subqueries == []
    assert result.expert_responses == []
    assert len(result.spans) == 1
    assert result.spans[0].attributes["subquery_count"] == 0
    assert result.final_response.startswith("No expert responses")


def test_agent_runtime_loads_recursion_config() -> None:
    config = AgentRuntimeConfig.from_files(ROOT)

    assert config.limits.max_depth == 4
    assert config.limits.max_breadth_per_agent == 6
    assert config.max_sibling_concurrency == 8


def test_agent_runtime_traces_selected_expert_and_context_counts() -> None:
    runtime = AgentRuntime.local(
        AgentRuntimeConfig(
            repo_root=ROOT,
            limits=ExecutionLimits(max_depth=2, max_breadth_per_agent=1),
        )
    )

    result = runtime.run(AgentRunRequest(query="How should CAL retrieve context?"))
    edl_spans = [span for span in result.spans if span.name == "edl.dispatch"]
    cal_spans = [span for span in result.spans if span.name == "cal.augment"]

    assert len(edl_spans) == 2
    assert edl_spans[0].attributes["expert_id"] == result.expert_responses[0].expert_id
    assert edl_spans[0].duration_ms is not None
    assert cal_spans[0].attributes["context_document_count"] >= 1


def test_agent_runtime_honors_total_instance_limit() -> None:
    runtime = AgentRuntime.local(
        AgentRuntimeConfig(
            repo_root=ROOT,
            limits=ExecutionLimits(
                max_depth=4,
                max_breadth_per_agent=3,
                max_total_agent_instances=2,
            ),
        )
    )

    result = runtime.run(AgentRunRequest(query="How should recursive execution stay bounded?"))
    skipped_spans = [span for span in result.spans if span.name == "agent.subquery.skipped"]

    assert len(result.subqueries) == 2
    assert len(result.expert_responses) == 2
    assert skipped_spans
    assert all(span.status == ExecutionStatus.CANCELLED for span in skipped_spans)


def test_recursion_guard_rejects_duplicate_query_signature() -> None:
    guard = RecursionGuard(ExecutionLimits(max_total_agent_instances=10))
    first = QueryEnvelope(
        sender_id="query:root",
        query_id="query:first",
        query="What context should CAL retrieve?",
    )
    duplicate = QueryEnvelope(
        sender_id="query:root",
        query_id="query:duplicate",
        query="what CONTEXT should cal retrieve",
    )

    assert guard.accept(first)
    assert not guard.accept(duplicate)


def test_agent_runtime_executes_sibling_subqueries_concurrently() -> None:
    class MeasuringCalTool:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = Lock()

        def send(
            self,
            subquery: QueryEnvelope,
            parent_context: ContextBundle,
        ) -> AugmentContextResponse:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            sleep(0.05)
            with self.lock:
                self.active -= 1
            return AugmentContextResponse(
                subquery=subquery.query,
                context=ContextBundle(query_id=subquery.query_id),
            )

    class FastEdlTool:
        def send(self, subquery: QueryEnvelope, context: ContextBundle) -> ExpertResponse:
            return ExpertResponse(
                sender_id=subquery.sender_id,
                query_id=subquery.query_id,
                subquery=subquery.query,
                expert_id="expert:test",
                response="ok",
                confidence=1.0,
            )

    cal_tool = MeasuringCalTool()
    runtime = AgentRuntime(
        config=AgentRuntimeConfig(
            repo_root=ROOT,
            limits=ExecutionLimits(max_depth=1, max_breadth_per_agent=3),
            max_sibling_concurrency=3,
        ),
        planner=DeterministicSubqueryGenerator(),
        cal_tool=cal_tool,
        edl_tool=FastEdlTool(),
        aggregator=ResponseAggregator(),
    )

    result = runtime.run(AgentRunRequest(query="Can siblings run concurrently?"))

    assert len(result.expert_responses) == 3
    assert cal_tool.max_active > 1
    assert result.spans[0].attributes["max_sibling_concurrency"] == 3


def test_agent_runtime_uses_batch_tools_for_sibling_subqueries() -> None:
    class BatchCalTool:
        def __init__(self) -> None:
            self.batch_calls: list[list[tuple[QueryEnvelope, ContextBundle]]] = []

        def send(
            self,
            subquery: QueryEnvelope,
            parent_context: ContextBundle,
        ) -> AugmentContextResponse:
            raise AssertionError("runtime should use send_batch when available")

        def send_batch(
            self,
            items: list[tuple[QueryEnvelope, ContextBundle]],
        ) -> list[AugmentContextResponse]:
            self.batch_calls.append(items)
            return [
                AugmentContextResponse(
                    subquery=subquery.query,
                    context=ContextBundle(query_id=subquery.query_id),
                )
                for subquery, _parent_context in items
            ]

    class BatchEdlTool:
        def __init__(self) -> None:
            self.batch_calls: list[list[tuple[QueryEnvelope, ContextBundle]]] = []

        def send(self, subquery: QueryEnvelope, context: ContextBundle) -> ExpertResponse:
            raise AssertionError("runtime should use send_batch when available")

        def send_batch(
            self,
            items: list[tuple[QueryEnvelope, ContextBundle]],
        ) -> list[ExpertResponse]:
            self.batch_calls.append(items)
            return [
                ExpertResponse(
                    sender_id=subquery.sender_id,
                    query_id=subquery.query_id,
                    subquery=subquery.query,
                    expert_id="expert:batch",
                    response=f"batch response for {subquery.query_id}",
                    confidence=1.0,
                )
                for subquery, _context in items
            ]

    cal_tool = BatchCalTool()
    edl_tool = BatchEdlTool()
    runtime = AgentRuntime(
        config=AgentRuntimeConfig(
            repo_root=ROOT,
            limits=ExecutionLimits(max_depth=1, max_breadth_per_agent=3),
        ),
        planner=DeterministicSubqueryGenerator(),
        cal_tool=cal_tool,
        edl_tool=edl_tool,
        aggregator=ResponseAggregator(),
    )

    result = runtime.run(AgentRunRequest(query="Can sibling queries batch through CAL and EDL?"))

    assert len(cal_tool.batch_calls) == 1
    assert len(edl_tool.batch_calls) == 1
    assert len(cal_tool.batch_calls[0]) == 3
    assert len(edl_tool.batch_calls[0]) == 3
    assert len(result.expert_responses) == 3
    assert all(response.expert_id == "expert:batch" for response in result.expert_responses)
    assert all(span.attributes.get("batch") is True for span in result.spans if span.name == "cal.augment")
    assert all(span.attributes.get("batch") is True for span in result.spans if span.name == "edl.dispatch")


def test_agent_runtime_marks_timed_out_subquery() -> None:
    class SlowCalTool:
        def send(
            self,
            subquery: QueryEnvelope,
            parent_context: ContextBundle,
        ) -> AugmentContextResponse:
            sleep(1.2)
            return AugmentContextResponse(
                subquery=subquery.query,
                context=ContextBundle(query_id=subquery.query_id),
            )

    class FastEdlTool:
        def send(self, subquery: QueryEnvelope, context: ContextBundle) -> ExpertResponse:
            return ExpertResponse(
                sender_id=subquery.sender_id,
                query_id=subquery.query_id,
                subquery=subquery.query,
                expert_id="expert:test",
                response="ok",
                confidence=1.0,
            )

    runtime = AgentRuntime(
        config=AgentRuntimeConfig(
            repo_root=ROOT,
            limits=ExecutionLimits(
                max_depth=1,
                max_breadth_per_agent=1,
                timeout_seconds_per_instance=1,
            ),
            max_sibling_concurrency=1,
        ),
        planner=DeterministicSubqueryGenerator(),
        cal_tool=SlowCalTool(),
        edl_tool=FastEdlTool(),
        aggregator=ResponseAggregator(),
    )

    started_at = perf_counter()
    result = runtime.run(AgentRunRequest(query="Will this timeout?"))
    elapsed = perf_counter() - started_at
    timeout_spans = [span for span in result.spans if span.name == "agent.subquery.timeout"]

    assert elapsed < 1.15
    assert result.expert_responses == []
    assert timeout_spans
    assert timeout_spans[0].status == ExecutionStatus.FAILED
    assert timeout_spans[0].attributes["timeout_seconds"] == 1
