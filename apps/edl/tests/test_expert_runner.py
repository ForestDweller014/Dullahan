from __future__ import annotations

import json

import edl.execution.model_provider as model_provider_module
import pytest
from dullahan_shared.schemas.context import ContextBundle, ContextDocument, ContextSource
from dullahan_shared.schemas.expert import ExpertProfile
from edl.api.schemas import DispatchRequest
from edl.config import EdlConfig
from edl.dispatch.attention_router import ExpertAttentionScore, ExpertRoute
from edl.execution.expert_runner import ExpertRunner
from edl.execution.model_provider import (
    ModelProvider,
    ModelRequest,
    ModelResult,
    OpenAICompatibleHttpProvider,
)
from edl.execution.prompt import ExpertPromptBuilder
from edl.service import ExpertDispatchService


class StubModelProvider(ModelProvider):
    def complete(self, request: ModelRequest) -> ModelResult:
        return ModelResult(
            text=f"Grounded test response for {request.model}",
            provider="stub-model",
            token_count=5,
        )


# Verifies that expert runner builds prompt and records model metadata.
def test_expert_runner_builds_prompt_and_records_model_metadata() -> None:
    expert = ExpertProfile(
        id="expert:test",
        cluster_id="cluster:test",
        role_context="You are responsible for context memory.",
        model="local-slm-test",
    )
    route = ExpertRoute(
        expert=expert,
        score=0.8,
        probability=0.7,
        distribution=[
            ExpertAttentionScore(
                expert_id=expert.id,
                raw_score=0.8,
                probability=0.7,
            )
        ],
    )
    request = DispatchRequest(
        sender_id="query:root",
        query_id="query:child",
        subquery="How should CAL retrieve context?",
        context=ContextBundle(
            query_id="query:child",
            documents=[
                ContextDocument(
                    id="doc:cal",
                    source=ContextSource.WORLD_STATE,
                    text="CAL retrieves parent and world-state context.",
                )
            ],
        ),
    )

    response = ExpertRunner(
        prompt_builder=ExpertPromptBuilder(),
        model_provider=StubModelProvider(),
    ).run(request, expert, route)

    assert response.expert_id == "expert:test"
    assert response.confidence == 0.7
    assert response.cited_context_document_ids == ["doc:cal"]
    assert response.routing_metadata["model"] == "local-slm-test"
    assert response.routing_metadata["model_provider"] == "stub-model"
    assert response.response == "Grounded test response for local-slm-test"


def test_expert_runner_overrides_local_expert_alias_for_hosted_model() -> None:
    expert = ExpertProfile(
        id="expert:test",
        cluster_id="cluster:test",
        role_context="You are a specialist.",
        model="local-slm-test",
    )
    route = ExpertRoute(expert=expert, score=1.0, probability=1.0, distribution=[])
    request = DispatchRequest(
        sender_id="query:root",
        query_id="query:child",
        subquery="Answer this",
        context=ContextBundle(query_id="query:child"),
    )

    response = ExpertRunner(
        prompt_builder=ExpertPromptBuilder(),
        model_provider=StubModelProvider(),
        model_override="gpt-5-mini",
    ).run(request, expert, route)

    assert response.response == "Grounded test response for gpt-5-mini"
    assert response.routing_metadata["model"] == "gpt-5-mini"
    assert response.routing_metadata["expert_model"] == "local-slm-test"


class FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


# Verifies that openai compatible HTTP provider posts completion request.
def test_openai_compatible_http_provider_posts_completion_request(monkeypatch) -> None:
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
                "choices": [{"text": "Expert model response"}],
                "usage": {"completion_tokens": 3},
            }
        )

    monkeypatch.setattr(model_provider_module, "urlopen", fake_urlopen)

    result = OpenAICompatibleHttpProvider(
        base_url="http://model.local/v1",
        timeout_seconds=4,
    ).complete(
        model_provider_module.ModelRequest(
            model="local-slm-test",
            prompt="Subquery:\nWhat now?",
            max_tokens=32,
        )
    )

    assert result.text == "Expert model response"
    assert result.provider == "openai-compatible-http"
    assert result.token_count == 3
    assert requests[0]["url"] == "http://model.local/v1/completions"
    assert requests[0]["payload"]["model"] == "local-slm-test"


# Verifies that EDL rejects missing native usage instead of faking a word-based count.
def test_openai_compatible_http_provider_requires_native_token_usage(monkeypatch) -> None:
    monkeypatch.setattr(
        model_provider_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeHttpResponse(
            {"choices": [{"text": "Expert model response"}]}
        ),
    )
    provider = OpenAICompatibleHttpProvider(base_url="http://model.local/v1")

    with pytest.raises(
        model_provider_module.ModelProviderError,
        match="native completion token usage",
    ):
        provider.complete(
            ModelRequest(model="local-slm-test", prompt="prompt", max_tokens=8)
        )


# Verifies that EDL config selects HTTP model provider.
def test_edl_config_selects_http_model_provider() -> None:
    provider = ExpertDispatchService._build_model_provider(
        EdlConfig(
            model_provider="http",
            model_base_url="http://model.local/v1",
            model_timeout_seconds=5,
        )
    )

    assert isinstance(provider, OpenAICompatibleHttpProvider)


# Verifies that EDL config rejects unknown model provider.
def test_edl_config_rejects_unknown_model_provider() -> None:
    with pytest.raises(ValueError, match="Input should be 'http' or 'openai'"):
        EdlConfig(model_provider="bogus")
