from __future__ import annotations

from cal.config import CalConfig
from cal.service import ContextAugmentationService
from dullahan_shared.tokenization import InferenceTokenCounter
from pydantic import SecretStr


def test_cal_openai_mode_keeps_dullahan_tokenize_boundary(tmp_path) -> None:
    service = ContextAugmentationService.from_config(
        CalConfig(
            repo_root=tmp_path,
            inference_provider="openai",
            inference_base_url="https://api.openai.com/v1",
            inference_api_key=SecretStr("test-key"),
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1024,
            tokenizer_model="Qwen/Qwen3-8B",
            tokenizer_base_url="http://tokenizer.local/v1",
        )
    )

    assert isinstance(service.token_counter, InferenceTokenCounter)
    assert service.token_counter.endpoint == "http://tokenizer.local/tokenize"
    assert service.world_state.embedding_model.request_dimensions is True


def test_cal_openai_environment_sets_hosted_models(monkeypatch) -> None:
    monkeypatch.setenv("DULLAHAN_INFERENCE_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    config = CalConfig.from_env()

    assert config.inference_base_url == "https://api.openai.com/v1"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.tokenizer_model == "Qwen/Qwen3-8B"
    assert config.tokenizer_base_url == "http://127.0.0.1:30000/v1"
    assert config.inference_api_key is not None
    assert config.inference_api_key.get_secret_value() == "test-key"
