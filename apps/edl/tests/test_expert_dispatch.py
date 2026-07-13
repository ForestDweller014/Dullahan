from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep

import pytest

from dullahan_shared.schemas.context import ContextBundle, ContextDocument, ContextSource
from dullahan_shared.schemas.expert import ExpertProfile, ExpertResponse
from edl.api.schemas import BatchDispatchRequest, DispatchRequest
from edl.config import EdlConfig
from edl.dispatch.attention_router import AttentionRouter
from edl.dispatch.expert_registry import ExpertRegistry
from edl.execution.model_provider import ModelProvider, ModelRequest, ModelResult
from edl.service import ExpertDispatchService

from testing_fakes import KeywordEmbeddingModel

ROOT = Path(__file__).resolve().parents[3]


class StubModelProvider(ModelProvider):
    def complete(self, request: ModelRequest) -> ModelResult:
        return ModelResult(
            text=f"Test expert response from {request.model}",
            provider="stub-model",
            token_count=5,
        )


def build_service() -> ExpertDispatchService:
    return ExpertDispatchService.from_config(
        EdlConfig(repo_root=ROOT),
        model_provider=StubModelProvider(),
        embedding_model=KeywordEmbeddingModel(),
    )


# Verifies that expert registry loads role contexts.
def test_expert_registry_loads_role_contexts() -> None:
    experts = ExpertRegistry(
        repo_root=ROOT,
        experts_path=ROOT / "memory" / "graph" / "experts.yaml",
    ).load()

    assert {expert.id for expert in experts} >= {"expert:context_memory", "expert:expert_dispatch"}
    assert all(expert.role_context for expert in experts)


# Verifies EDL routing behavior with inference embeddings explicitly mocked.
def test_edl_routes_context_query_to_context_memory_expert() -> None:
    request = DispatchRequest(
        sender_id="agent:root",
        query_id="query:cal",
        subquery="How should CAL retrieve world-state context and pack token budgets?",
        context=ContextBundle(
            query_id="query:cal",
            documents=[
                ContextDocument(
                    id="doc:cal",
                    source=ContextSource.GRAPH_NODE,
                    text="CAL retrieves parent context and world-state context.",
                )
            ],
        ),
    )

    response = build_service().dispatch(request).response

    assert response.expert_id == "expert:context_memory"
    assert response.sender_id == "agent:root"
    assert response.cited_context_document_ids == ["doc:cal"]
    assert response.routing_metadata["candidate_count"] >= 1
    assert response.routing_metadata["route_probability"] == response.confidence
    assert response.routing_metadata["attention_scoring"] == "embedding_cosine"
    assert response.routing_metadata["model_provider"] == "stub-model"
    assert "local-slm-context" in response.response
    assert not hasattr(response, "context")


# Verifies dispatch routing behavior with inference embeddings explicitly mocked.
def test_edl_routes_dispatch_query_to_dispatch_expert() -> None:
    request = DispatchRequest(
        sender_id="agent:root",
        query_id="query:edl",
        subquery="How does attention routing select an expert from the expert pool?",
        context=ContextBundle(query_id="query:edl", documents=[]),
    )

    response = build_service().dispatch(request).response

    assert response.expert_id == "expert:expert_dispatch"
    assert response.confidence is not None
    assert response.routing_metadata["model"] == "local-slm-dispatch"


# Verifies attention softmax behavior with inference embeddings explicitly mocked.
def test_attention_router_returns_softmax_distribution() -> None:
    experts = ExpertRegistry(
        repo_root=ROOT,
        experts_path=ROOT / "memory" / "graph" / "experts.yaml",
    ).load()

    route = AttentionRouter(embedding_model=KeywordEmbeddingModel()).select(
        "How should EDL route to the expert pool?",
        experts,
    )

    probabilities = [score.probability for score in route.distribution]

    assert route.expert.id == "expert:expert_dispatch"
    assert len(route.distribution) == len(experts)
    assert abs(sum(probabilities) - 1.0) < 0.00001
    assert route.probability == max(probabilities)


# Verifies batch ordering with inference embeddings and model completion explicitly mocked.
def test_edl_batch_dispatch_returns_responses_in_request_order() -> None:
    service = build_service()
    request = BatchDispatchRequest(
        requests=[
            DispatchRequest(
                sender_id="agent:root",
                query_id="query:cal",
                subquery="How should CAL retrieve context?",
                context=ContextBundle(query_id="query:cal"),
            ),
            DispatchRequest(
                sender_id="agent:root",
                query_id="query:edl",
                subquery="How should EDL select an expert?",
                context=ContextBundle(query_id="query:edl"),
            ),
        ]
    )

    response = service.dispatch_batch(request)

    assert [item.query_id for item in response.responses] == ["query:cal", "query:edl"]
    assert len(response.responses) == 2


# Verifies concurrent dispatch while inference embeddings and expert execution are mocked.
def test_edl_batch_dispatch_runs_expert_instances_concurrently() -> None:
    class MeasuringRunner:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = Lock()

        def run(self, request, expert, route):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            sleep(0.05)
            with self.lock:
                self.active -= 1
            return ExpertResponse(
                sender_id=request.sender_id,
                query_id=request.query_id,
                subquery=request.subquery,
                expert_id=expert.id,
                response="ok",
                confidence=route.probability,
            )

    runner = MeasuringRunner()
    service = ExpertDispatchService(
        registry=ExpertRegistry(
            repo_root=ROOT,
            experts_path=ROOT / "memory" / "graph" / "experts.yaml",
        ),
        router=AttentionRouter(embedding_model=KeywordEmbeddingModel()),
        runner=runner,
        max_dispatch_concurrency=3,
    )
    request = BatchDispatchRequest(
        requests=[
            DispatchRequest(
                sender_id="agent:root",
                query_id=f"query:{index}",
                subquery=f"How should EDL route query {index}?",
                context=ContextBundle(query_id=f"query:{index}"),
            )
            for index in range(3)
        ]
    )

    response = service.dispatch_batch(request)

    assert len(response.responses) == 3
    assert runner.max_active > 1


# Verifies an expert's configured concurrency limit applies across simultaneous dispatch calls.
def test_edl_enforces_expert_max_concurrency_across_dispatch_calls() -> None:
    expert = ExpertProfile(
        id="expert:limited",
        cluster_id="cluster:limited",
        role_context="Handles every request in this test.",
        model="limited-model",
        max_concurrency=2,
    )

    class StaticRegistry:
        def load(self) -> list[ExpertProfile]:
            return [expert]

    class MeasuringRunner:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = Lock()

        def run(self, request, expert, route):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            sleep(0.03)
            with self.lock:
                self.active -= 1
            return ExpertResponse(
                sender_id=request.sender_id,
                query_id=request.query_id,
                subquery=request.subquery,
                expert_id=expert.id,
                response="ok",
                confidence=route.probability,
            )

    runner = MeasuringRunner()
    service = ExpertDispatchService(
        registry=StaticRegistry(),
        router=AttentionRouter(embedding_model=KeywordEmbeddingModel()),
        runner=runner,
        max_dispatch_concurrency=8,
    )
    requests = [
        DispatchRequest(
            sender_id="agent:root",
            query_id=f"query:limited:{index}",
            subquery=f"Handle limited request {index}",
            context=ContextBundle(query_id=f"query:limited:{index}"),
        )
        for index in range(8)
    ]

    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        responses = list(executor.map(service.dispatch, requests))

    assert len(responses) == len(requests)
    assert runner.max_active == expert.max_concurrency


# Verifies a failed expert invocation releases its slot for later work.
def test_edl_releases_expert_concurrency_slot_after_failure() -> None:
    expert = ExpertProfile(
        id="expert:failure",
        cluster_id="cluster:failure",
        role_context="Fails once and then succeeds.",
        model="failure-model",
        max_concurrency=1,
    )

    class StaticRegistry:
        def load(self) -> list[ExpertProfile]:
            return [expert]

    class FailsOnceRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, request, expert, route):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("inference failed")
            return ExpertResponse(
                sender_id=request.sender_id,
                query_id=request.query_id,
                subquery=request.subquery,
                expert_id=expert.id,
                response="recovered",
                confidence=route.probability,
            )

    runner = FailsOnceRunner()
    service = ExpertDispatchService(
        registry=StaticRegistry(),
        router=AttentionRouter(embedding_model=KeywordEmbeddingModel()),
        runner=runner,
        max_dispatch_concurrency=2,
    )
    first = DispatchRequest(
        sender_id="agent:root",
        query_id="query:failure:1",
        subquery="Fail this request",
        context=ContextBundle(query_id="query:failure:1"),
    )
    second = DispatchRequest(
        sender_id="agent:root",
        query_id="query:failure:2",
        subquery="Recover this request",
        context=ContextBundle(query_id="query:failure:2"),
    )

    with pytest.raises(RuntimeError, match="inference failed"):
        service.dispatch(first)

    response = service.dispatch(second)

    assert response.response.response == "recovered"
    assert runner.calls == 2
