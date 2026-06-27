from __future__ import annotations

import json

import agent_runtime.planning.provider as provider_module
from agent_runtime.config import AgentRuntimeConfig
from agent_runtime.agent import AgentRuntime
from agent_runtime.planning.provider import (
    DeterministicPlannerProvider,
    OpenAICompatiblePlannerProvider,
    PlannerRequest,
)
from dullahan_shared.schemas.query import QueryEnvelope


class FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_deterministic_planner_provider_limits_subqueries() -> None:
    result = DeterministicPlannerProvider().plan(
        PlannerRequest(
            parent_query=QueryEnvelope(
                sender_id="user",
                query_id="query:root",
                query="How should CAL and EDL cooperate?",
            ),
            max_breadth=2,
        )
    )

    assert result.provider == "deterministic-planner"
    assert len(result.subqueries) == 2


def test_http_planner_provider_parses_line_based_subqueries(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeHttpResponse(
            {
                "choices": [
                    {
                        "text": "1. What context is needed?\n2. Which expert should answer?\n3. Ignore this"
                    }
                ]
            }
        )

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)

    result = OpenAICompatiblePlannerProvider(
        base_url="http://planner.local/v1",
        model="planner-model",
        timeout_seconds=6,
    ).plan(
        PlannerRequest(
            parent_query=QueryEnvelope(
                sender_id="user",
                query_id="query:root",
                query="How should the runtime plan?",
            ),
            max_breadth=2,
        )
    )

    assert result.provider == "openai-compatible-planner"
    assert result.subqueries == ["What context is needed?", "Which expert should answer?"]
    assert requests[0]["url"] == "http://planner.local/v1/completions"
    assert requests[0]["payload"]["model"] == "planner-model"


def test_agent_runtime_selects_http_planner_provider() -> None:
    provider = AgentRuntime._build_planner_provider(
        AgentRuntimeConfig(
            planner_provider="http",
            planner_base_url="http://planner.local/v1",
            planner_model="planner-model",
            planner_timeout_seconds=5,
        )
    )

    assert isinstance(provider, OpenAICompatiblePlannerProvider)
