from __future__ import annotations

import json

import agent_runtime.aggregation as aggregation_module
import pytest
from agent_runtime.aggregation import (
    OpenAICompatibleSynthesisProvider,
    ResponseAggregator,
    SynthesisProviderError,
)
from dullahan_shared.schemas.expert import ExpertResponse
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


# Verifies paired subquery-answer synthesis while the inference HTTP boundary is mocked.
def test_response_aggregator_sends_paired_subquery_answers_to_inference(monkeypatch) -> None:
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
                "choices": [{"text": "A coherent final answer."}],
                "usage": {"prompt_tokens": 91, "completion_tokens": 5},
            }
        )

    monkeypatch.setattr(aggregation_module, "urlopen", fake_urlopen)
    root_query = QueryEnvelope(
        sender_id="user",
        query_id="query:root",
        query="How should the migration proceed?",
    )
    responses = [
        ExpertResponse(
            sender_id="query:root",
            query_id="query:backup",
            subquery="How should backups be verified?",
            expert_id="expert:reliability",
            response="Restore a sample and compare checksums.",
        ),
        ExpertResponse(
            sender_id="query:root",
            query_id="query:cutover",
            subquery="How should reads be cut over?",
            expert_id="expert:database",
            response="Switch reads only after dual-write parity is stable.",
        ),
    ]
    aggregator = ResponseAggregator(
        provider=OpenAICompatibleSynthesisProvider(
            base_url="http://inference.local/v1",
            model="final-model",
            timeout_seconds=12,
        ),
        max_tokens=300,
    )

    result = aggregator.synthesize(root_query, responses)

    assert result.text == "A coherent final answer."
    assert result.prompt_tokens == 91
    assert result.completion_tokens == 5
    assert requests[0]["url"] == "http://inference.local/v1/completions"
    assert requests[0]["payload"]["model"] == "final-model"
    assert requests[0]["payload"]["max_tokens"] == 300
    prompt = requests[0]["payload"]["prompt"]
    for response in responses:
        assert response.query_id in prompt
        assert response.subquery in prompt
        assert response.expert_id in prompt
        assert response.response in prompt


# Verifies an empty expert result returns an explicit message without invoking inference.
def test_response_aggregator_does_not_invoke_inference_without_evidence(monkeypatch) -> None:
    def fail_urlopen(request, timeout):
        raise AssertionError("inference must not run without expert evidence")

    monkeypatch.setattr(aggregation_module, "urlopen", fail_urlopen)
    aggregator = ResponseAggregator(
        provider=OpenAICompatibleSynthesisProvider(
            base_url="http://inference.local/v1",
            model="final-model",
        )
    )
    root_query = QueryEnvelope(
        sender_id="user",
        query_id="query:root",
        query="Can this be answered?",
    )

    result = aggregator.synthesize(root_query, [])

    assert result.provider == "not-invoked"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.text == "No expert responses were produced for: Can this be answered?"


# Verifies final synthesis rejects mocked inference responses without native token usage.
def test_synthesis_provider_requires_native_token_usage(monkeypatch) -> None:
    monkeypatch.setattr(
        aggregation_module,
        "urlopen",
        lambda request, timeout: FakeHttpResponse(
            {"choices": [{"text": "An answer without usage metadata."}]}
        ),
    )
    aggregator = ResponseAggregator(
        provider=OpenAICompatibleSynthesisProvider(
            base_url="http://inference.local/v1",
            model="final-model",
        )
    )
    root_query = QueryEnvelope(
        sender_id="user",
        query_id="query:root",
        query="How should this be answered?",
    )
    responses = [
        ExpertResponse(
            sender_id="query:root",
            query_id="query:evidence",
            subquery="What evidence is available?",
            expert_id="expert:test",
            response="Verified evidence is available.",
        )
    ]

    with pytest.raises(
        SynthesisProviderError,
        match="no native prompt token usage",
    ):
        aggregator.synthesize(root_query, responses)


def test_openai_synthesis_maps_responses_usage(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        return FakeHttpResponse(
            {
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": "Hosted final answer."}
                        ]
                    }
                ],
                "usage": {"input_tokens": 20, "output_tokens": 4},
            }
        )

    monkeypatch.setattr(aggregation_module, "urlopen", fake_urlopen)
    result = OpenAICompatibleSynthesisProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-5-mini",
        api_mode="responses",
        api_key="test-key",
    ).synthesize(aggregation_module.SynthesisRequest(prompt="Evidence", max_tokens=100))

    assert result.text == "Hosted final answer."
    assert result.prompt_tokens == 20
    assert result.completion_tokens == 4
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["authorization"] == "Bearer test-key"
