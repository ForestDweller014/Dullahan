from __future__ import annotations

import json

import agent_runtime.planning.provider as provider_module
import pytest
from agent_runtime.agent import AgentRuntime
from agent_runtime.aggregation import OpenAICompatibleSynthesisProvider
from agent_runtime.config import AgentRuntimeConfig
from agent_runtime.planning.provider import (
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


# Verifies that HTTP planner provider parses line based subqueries.
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
                        "text": (
                            "1. What context is needed?\n"
                            "2. Which expert should answer?\n"
                            "3. Ignore this"
                        )
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
    assert requests[0]["payload"]["temperature"] == 0


def test_openai_planner_uses_responses_endpoint_and_bearer_auth(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["authorization"] = request.get_header("Authorization")
        return FakeHttpResponse({"output_text": "First question\nSecond question"})

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    result = OpenAICompatiblePlannerProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-5-mini",
        api_mode="responses",
        api_key="test-key",
    ).plan(
        PlannerRequest(
            parent_query=QueryEnvelope(
                sender_id="user",
                query_id="query:root",
                query="Plan this",
            ),
            max_breadth=2,
        )
    )

    assert result.subqueries == ["First question", "Second question"]
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["payload"]["model"] == "gpt-5-mini"
    assert "input" in captured["payload"]
    assert "prompt" not in captured["payload"]
    assert captured["authorization"] == "Bearer test-key"


# Verifies that agent runtime selects HTTP planner provider.
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


# Verifies that the runtime defaults to the OpenAI-compatible HTTP planner.
def test_agent_runtime_defaults_to_http_planner_provider() -> None:
    provider = AgentRuntime._build_planner_provider(AgentRuntimeConfig())

    assert isinstance(provider, OpenAICompatiblePlannerProvider)


# Verifies that agent runtime builds an OpenAI-compatible final synthesis provider.
def test_agent_runtime_selects_http_synthesis_provider() -> None:
    provider = AgentRuntime._build_synthesis_provider(
        AgentRuntimeConfig(
            synthesis_provider="http",
            synthesis_base_url="http://synthesis.local/v1",
            synthesis_model="final-model",
            synthesis_timeout_seconds=9,
        )
    )

    assert isinstance(provider, OpenAICompatibleSynthesisProvider)
    assert provider.base_url == "http://synthesis.local/v1"
    assert provider.model == "final-model"
    assert provider.timeout_seconds == 9


# Verifies that planner and synthesis inference settings come from the environment.
def test_agent_runtime_config_reads_planner_environment(monkeypatch, tmp_path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "recursion.yaml").write_text("max_depth: 1\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_PLANNER_PROVIDER", "http")
    monkeypatch.setenv("AGENT_PLANNER_MODEL", "local-planner")
    monkeypatch.setenv("AGENT_PLANNER_BASE_URL", "http://inference.local/v1")
    monkeypatch.setenv("AGENT_PLANNER_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("AGENT_SYNTHESIS_PROVIDER", "http")
    monkeypatch.setenv("AGENT_SYNTHESIS_MODEL", "final-model")
    monkeypatch.setenv("AGENT_SYNTHESIS_BASE_URL", "http://synthesis.local/v1")
    monkeypatch.setenv("AGENT_SYNTHESIS_TIMEOUT_SECONDS", "75")
    monkeypatch.setenv("AGENT_SYNTHESIS_MAX_TOKENS", "700")

    config = AgentRuntimeConfig.from_files(tmp_path)

    assert config.planner_provider == "http"
    assert config.planner_base_url == "http://inference.local/v1"
    assert config.planner_timeout_seconds == 45
    assert config.synthesis_provider == "http"
    assert config.synthesis_model == "final-model"
    assert config.synthesis_base_url == "http://synthesis.local/v1"
    assert config.synthesis_timeout_seconds == 75
    assert config.synthesis_max_tokens == 700


def test_agent_runtime_config_uses_shared_openai_environment(monkeypatch, tmp_path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "recursion.yaml").write_text("max_depth: 1\n", encoding="utf-8")
    monkeypatch.setenv("DULLAHAN_INFERENCE_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    config = AgentRuntimeConfig.from_files(tmp_path)

    assert config.planner_provider == "openai"
    assert config.synthesis_provider == "openai"
    assert config.planner_base_url == "https://api.openai.com/v1"
    assert config.planner_model == "gpt-test"
    assert config.synthesis_model == "gpt-test"
    assert config.planner_api_key is not None
    assert config.planner_api_key.get_secret_value() == "test-key"


def test_openai_configuration_requires_bearer_token(monkeypatch, tmp_path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "recursion.yaml").write_text("max_depth: 1\n", encoding="utf-8")
    monkeypatch.setenv("DULLAHAN_INFERENCE_PROVIDER", "openai")
    monkeypatch.delenv("DULLAHAN_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        AgentRuntimeConfig.from_files(tmp_path)
