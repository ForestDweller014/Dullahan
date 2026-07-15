from __future__ import annotations

from edl.config import EdlConfig
from edl.service import ExpertDispatchService


def test_edl_openai_environment_replaces_local_expert_model_alias(monkeypatch) -> None:
    monkeypatch.setenv("DULLAHAN_INFERENCE_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    config = EdlConfig.from_env()
    provider = ExpertDispatchService._build_model_provider(config)

    assert config.model_provider == "openai"
    assert config.model_base_url == "https://api.openai.com/v1"
    assert config.model_override == "gpt-test"
    assert config.embedding_model == "text-embedding-3-small"
    assert provider.api_mode == "responses"
    assert provider.api_key == "test-key"
