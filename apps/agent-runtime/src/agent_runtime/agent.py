from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError

from cal.config import CalConfig
from cal.service import ContextAugmentationService
from dullahan_shared.embeddings import EmbeddingModel
from dullahan_shared.ids import new_id
from dullahan_shared.inference import provider_api_mode
from dullahan_shared.schemas.context import ContextBundle
from dullahan_shared.schemas.execution import ExecutionStatus
from dullahan_shared.schemas.expert import ExpertResponse
from dullahan_shared.schemas.query import QueryEnvelope
from dullahan_shared.tokenization import TokenCounter
from edl.config import EdlConfig
from edl.execution.model_provider import ModelProvider
from edl.service import ExpertDispatchService

from agent_runtime.aggregation import (
    OpenAICompatibleSynthesisProvider,
    ResponseAggregator,
    SynthesisProvider,
)
from agent_runtime.artifacts import ExecutionArtifactStore
from agent_runtime.collections import ThreadSafeList
from agent_runtime.config import AgentRuntimeConfig
from agent_runtime.models import AgentRunRequest, AgentRunResult
from agent_runtime.planning.provider import (
    OpenAICompatiblePlannerProvider,
    PlannerProvider,
)
from agent_runtime.planning.subquery_generator import SubqueryGenerator
from agent_runtime.recursion import RecursionGuard
from agent_runtime.tools.http_cal import HttpCalTool
from agent_runtime.tools.http_edl import HttpEdlTool
from agent_runtime.tools.local_cal import LocalCalTool
from agent_runtime.tools.local_edl import LocalEdlTool
from agent_runtime.tracing import InMemoryTraceCollector


class AgentRuntime:
    def __init__(
        self,
        *,
        config: AgentRuntimeConfig,
        planner: SubqueryGenerator,
        cal_tool: LocalCalTool,
        edl_tool: LocalEdlTool,
        aggregator: ResponseAggregator,
    ) -> None:
        self.config = config
        self.planner = planner
        self.cal_tool = cal_tool
        self.edl_tool = edl_tool
        self.aggregator = aggregator

    @classmethod
    def local(
        cls,
        config: AgentRuntimeConfig,
        *,
        planner_provider: PlannerProvider | None = None,
        model_provider: ModelProvider | None = None,
        embedding_model: EmbeddingModel | None = None,
        token_counter: TokenCounter | None = None,
        cal_config: CalConfig | None = None,
        synthesis_provider: SynthesisProvider | None = None,
    ) -> AgentRuntime:
        edl_config = EdlConfig.from_env().model_copy(update={"repo_root": config.repo_root})
        resolved_cal_config = cal_config or CalConfig.from_env().model_copy(
            update={"repo_root": config.repo_root}
        )
        return cls(
            config=config,
            planner=SubqueryGenerator(planner_provider or cls._build_planner_provider(config)),
            cal_tool=LocalCalTool(
                ContextAugmentationService.from_config(
                    resolved_cal_config,
                    embedding_model=embedding_model,
                    token_counter=token_counter,
                )
            ),
            edl_tool=LocalEdlTool(
                ExpertDispatchService.from_config(
                    edl_config,
                    model_provider=model_provider,
                    embedding_model=embedding_model,
                )
            ),
            aggregator=ResponseAggregator(
                provider=synthesis_provider or cls._build_synthesis_provider(config),
                max_tokens=config.synthesis_max_tokens,
            ),
        )

    @classmethod
    def remote(
        cls,
        *,
        config: AgentRuntimeConfig,
        cal_base_url: str,
        edl_base_url: str,
        timeout_seconds: float = 30.0,
        synthesis_provider: SynthesisProvider | None = None,
    ) -> AgentRuntime:
        return cls(
            config=config,
            planner=SubqueryGenerator(cls._build_planner_provider(config)),
            cal_tool=HttpCalTool(cal_base_url, timeout_seconds=timeout_seconds),
            edl_tool=HttpEdlTool(edl_base_url, timeout_seconds=timeout_seconds),
            aggregator=ResponseAggregator(
                provider=synthesis_provider or cls._build_synthesis_provider(config),
                max_tokens=config.synthesis_max_tokens,
            ),
        )

    @staticmethod
    def _build_planner_provider(config: AgentRuntimeConfig) -> PlannerProvider:
        return OpenAICompatiblePlannerProvider(
            base_url=config.planner_base_url,
            model=config.planner_model,
            timeout_seconds=config.planner_timeout_seconds,
            api_mode=provider_api_mode(config.planner_provider),
            api_key=(
                config.planner_api_key.get_secret_value() if config.planner_api_key else None
            ),
        )

    @staticmethod
    def _build_synthesis_provider(config: AgentRuntimeConfig) -> SynthesisProvider:
        return OpenAICompatibleSynthesisProvider(
            base_url=config.synthesis_base_url,
            model=config.synthesis_model,
            timeout_seconds=config.synthesis_timeout_seconds,
            api_mode=provider_api_mode(config.synthesis_provider),
            api_key=(
                config.synthesis_api_key.get_secret_value()
                if config.synthesis_api_key
                else None
            ),
        )

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        trace = InMemoryTraceCollector()
        guard = RecursionGuard(self.config.limits)
        root_query = QueryEnvelope(
            sender_id=request.sender_id,
            query_id=new_id("query"),
            query=request.query,
            parent_context=ContextBundle(query_id="root"),
            depth=0,
        )
        root_span = trace.start_span(
            name="agent.run",
            query_id=root_query.query_id,
            depth=root_query.depth,
            attributes={
                "sender_id": request.sender_id,
                "max_sibling_concurrency": self.config.max_sibling_concurrency,
            },
        )
        subqueries: list[QueryEnvelope] = []
        contexts: ThreadSafeList[ContextBundle] = ThreadSafeList()
        responses = self._execute_children(
            parent_query=root_query,
            parent_context=root_query.parent_context or ContextBundle(query_id=root_query.query_id),
            parent_span_id=root_span.span_id,
            trace=trace,
            guard=guard,
            collected_subqueries=subqueries,
            collected_contexts=contexts,
        )

        synthesis_span = trace.start_span(
            name="agent.synthesis",
            query_id=root_query.query_id,
            parent_span_id=root_span.span_id,
            depth=root_query.depth,
            attributes={"expert_response_count": len(responses)},
        )
        synthesis = self.aggregator.synthesize(root_query, responses)
        trace.end_span(
            synthesis_span,
            attributes={
                "provider": synthesis.provider,
                "prompt_token_count": synthesis.prompt_tokens,
                "completion_token_count": synthesis.completion_tokens,
            },
        )

        trace.end_span(
            root_span,
            status=ExecutionStatus.SUCCEEDED,
            attributes={
                "subquery_count": len(subqueries),
                "expert_response_count": len(responses),
            },
        )

        result = AgentRunResult(
            root_query=root_query,
            subqueries=subqueries,
            contexts=contexts.snapshot(),
            expert_responses=responses,
            trace_id=trace.trace_id,
            spans=trace.spans,
            final_response=synthesis.text,
        )
        if request.persist_artifacts:
            artifact_dir = ExecutionArtifactStore(
                self.config.repo_root / "memory" / "executions"
            ).write_run(result)
            result = result.model_copy(update={"artifact_dir": str(artifact_dir)})
        return result

    def _execute_children(
        self,
        *,
        parent_query: QueryEnvelope,
        parent_context: ContextBundle,
        parent_span_id: str,
        trace: InMemoryTraceCollector,
        guard: RecursionGuard,
        collected_subqueries: list[QueryEnvelope],
        collected_contexts: ThreadSafeList[ContextBundle],
    ) -> list[ExpertResponse]:
        if not guard.can_generate_children(parent_query):
            return []

        generated = self.planner.generate(
            parent_query,
            max_breadth=self.config.limits.max_breadth_per_agent,
        )

        accepted_subqueries: list[QueryEnvelope] = []
        for subquery in generated:
            if not guard.accept(subquery):
                skipped_span = trace.start_span(
                    name="agent.subquery.skipped",
                    query_id=subquery.query_id,
                    parent_span_id=parent_span_id,
                    parent_query_id=parent_query.query_id,
                    depth=subquery.depth,
                    attributes={
                        "subquery": subquery.query,
                        "reason": "duplicate_or_instance_limit",
                    },
                )
                trace.end_span(skipped_span, status=ExecutionStatus.CANCELLED)
                continue

            collected_subqueries.append(subquery)
            accepted_subqueries.append(subquery)

        if not accepted_subqueries:
            return []

        if self._can_batch_dispatch():
            return self._execute_batch_subqueries(
                accepted_subqueries=accepted_subqueries,
                parent_query=parent_query,
                parent_context=parent_context,
                parent_span_id=parent_span_id,
                trace=trace,
                guard=guard,
                collected_subqueries=collected_subqueries,
                collected_contexts=collected_contexts,
            )

        worker_count = min(self.config.max_sibling_concurrency, len(accepted_subqueries))
        executor = ThreadPoolExecutor(max_workers=worker_count)
        future_by_subquery = {
            executor.submit(
                self._execute_single_subquery,
                subquery=subquery,
                parent_query=parent_query,
                parent_context=parent_context,
                parent_span_id=parent_span_id,
                trace=trace,
                guard=guard,
                collected_subqueries=collected_subqueries,
                collected_contexts=collected_contexts,
            ): subquery
            for subquery in accepted_subqueries
        }

        responses: list[ExpertResponse] = []
        try:
            for future, subquery in future_by_subquery.items():
                timeout_seconds = self.config.limits.timeout_seconds_per_instance
                try:
                    responses.extend(future.result(timeout=timeout_seconds))
                except TimeoutError:
                    future.cancel()
                    timeout_span = trace.start_span(
                        name="agent.subquery.timeout",
                        query_id=subquery.query_id,
                        parent_span_id=parent_span_id,
                        parent_query_id=parent_query.query_id,
                        depth=subquery.depth,
                        attributes={
                            "subquery": subquery.query,
                            "timeout_seconds": timeout_seconds,
                        },
                    )
                    trace.end_span(timeout_span, status=ExecutionStatus.FAILED)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return responses

    def _can_batch_dispatch(self) -> bool:
        return callable(getattr(self.cal_tool, "send_batch", None)) and callable(
            getattr(self.edl_tool, "send_batch", None)
        )

    def _execute_batch_subqueries(
        self,
        *,
        accepted_subqueries: list[QueryEnvelope],
        parent_query: QueryEnvelope,
        parent_context: ContextBundle,
        parent_span_id: str,
        trace: InMemoryTraceCollector,
        guard: RecursionGuard,
        collected_subqueries: list[QueryEnvelope],
        collected_contexts: ThreadSafeList[ContextBundle],
    ) -> list[ExpertResponse]:
        subquery_spans = {
            subquery.query_id: trace.start_span(
                name="agent.subquery",
                query_id=subquery.query_id,
                parent_span_id=parent_span_id,
                parent_query_id=parent_query.query_id,
                depth=subquery.depth,
                attributes={"subquery": subquery.query, "batch": True},
            )
            for subquery in accepted_subqueries
        }
        cal_spans = {
            subquery.query_id: trace.start_span(
                name="cal.augment",
                query_id=subquery.query_id,
                parent_span_id=subquery_spans[subquery.query_id].span_id,
                parent_query_id=parent_query.query_id,
                depth=subquery.depth,
                attributes={"batch": True},
            )
            for subquery in accepted_subqueries
        }

        augmented_responses = self.cal_tool.send_batch(
            [(subquery, parent_context) for subquery in accepted_subqueries]
        )
        if len(augmented_responses) != len(accepted_subqueries):
            raise RuntimeError(
                "CAL batch response count did not match submitted subquery count: "
                f"{len(augmented_responses)} != {len(accepted_subqueries)}"
            )

        augmented_by_query_id = {}
        for subquery, augmented in zip(accepted_subqueries, augmented_responses, strict=True):
            augmented_by_query_id[subquery.query_id] = augmented
            collected_contexts.append(augmented.context)
            trace.end_span(
                cal_spans[subquery.query_id],
                attributes={
                    "batch": True,
                    "context_document_count": len(augmented.context.documents),
                },
            )

        edl_spans = {
            subquery.query_id: trace.start_span(
                name="edl.dispatch",
                query_id=subquery.query_id,
                parent_span_id=subquery_spans[subquery.query_id].span_id,
                parent_query_id=parent_query.query_id,
                depth=subquery.depth,
                attributes={"batch": True},
            )
            for subquery in accepted_subqueries
        }
        expert_responses = self.edl_tool.send_batch(
            [
                (subquery, augmented_by_query_id[subquery.query_id].context)
                for subquery in accepted_subqueries
            ]
        )
        if len(expert_responses) != len(accepted_subqueries):
            raise RuntimeError(
                "EDL batch response count did not match submitted subquery count: "
                f"{len(expert_responses)} != {len(accepted_subqueries)}"
            )

        responses: list[ExpertResponse] = []
        for subquery, response in zip(accepted_subqueries, expert_responses, strict=True):
            trace.end_span(
                edl_spans[subquery.query_id],
                attributes={
                    "batch": True,
                    "expert_id": response.expert_id,
                    "confidence": response.confidence or 0.0,
                    "cited_context_document_count": len(response.cited_context_document_ids),
                },
            )
            responses.append(response)

        for subquery, response in zip(accepted_subqueries, expert_responses, strict=True):
            augmented = augmented_by_query_id[subquery.query_id]
            child_responses = self._execute_children(
                parent_query=subquery,
                parent_context=augmented.context,
                parent_span_id=subquery_spans[subquery.query_id].span_id,
                trace=trace,
                guard=guard,
                collected_subqueries=collected_subqueries,
                collected_contexts=collected_contexts,
            )
            responses.extend(child_responses)
            trace.end_span(
                subquery_spans[subquery.query_id],
                attributes={
                    "batch": True,
                    "expert_id": response.expert_id,
                    "response_confidence": response.confidence or 0.0,
                    "child_response_count": len(child_responses),
                },
            )

        return responses

    def _execute_single_subquery(
        self,
        *,
        subquery: QueryEnvelope,
        parent_query: QueryEnvelope,
        parent_context: ContextBundle,
        parent_span_id: str,
        trace: InMemoryTraceCollector,
        guard: RecursionGuard,
        collected_subqueries: list[QueryEnvelope],
        collected_contexts: ThreadSafeList[ContextBundle],
    ) -> list[ExpertResponse]:
        responses: list[ExpertResponse] = []
        subquery_span = trace.start_span(
            name="agent.subquery",
            query_id=subquery.query_id,
            parent_span_id=parent_span_id,
            parent_query_id=parent_query.query_id,
            depth=subquery.depth,
            attributes={"subquery": subquery.query},
        )
        cal_span = trace.start_span(
            name="cal.augment",
            query_id=subquery.query_id,
            parent_span_id=subquery_span.span_id,
            parent_query_id=parent_query.query_id,
            depth=subquery.depth,
        )
        augmented = self.cal_tool.send(subquery=subquery, parent_context=parent_context)
        collected_contexts.append(augmented.context)
        trace.end_span(
            cal_span,
            attributes={"context_document_count": len(augmented.context.documents)},
        )
        edl_span = trace.start_span(
            name="edl.dispatch",
            query_id=subquery.query_id,
            parent_span_id=subquery_span.span_id,
            parent_query_id=parent_query.query_id,
            depth=subquery.depth,
        )
        response = self.edl_tool.send(subquery=subquery, context=augmented.context)
        trace.end_span(
            edl_span,
            attributes={
                "expert_id": response.expert_id,
                "confidence": response.confidence or 0.0,
                "cited_context_document_count": len(response.cited_context_document_ids),
            },
        )
        responses.append(response)
        child_responses = self._execute_children(
            parent_query=subquery,
            parent_context=augmented.context,
            parent_span_id=subquery_span.span_id,
            trace=trace,
            guard=guard,
            collected_subqueries=collected_subqueries,
            collected_contexts=collected_contexts,
        )
        responses.extend(child_responses)
        trace.end_span(
            subquery_span,
            attributes={
                "expert_id": response.expert_id,
                "response_confidence": response.confidence or 0.0,
                "child_response_count": len(child_responses),
            },
        )

        return responses
