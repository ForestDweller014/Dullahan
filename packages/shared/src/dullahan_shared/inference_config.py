from __future__ import annotations

import os

from dullahan_shared.inference import InferenceProvider

LOCAL_INFERENCE_BASE_URL = "http://127.0.0.1:30000/v1"
OPENAI_INFERENCE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_LOCAL_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LOCAL_TOKENIZER_MODEL = "Qwen/Qwen3-8B"


def inference_provider(specific_name: str | None = None) -> InferenceProvider:
    value = os.getenv(specific_name) if specific_name else None
    provider = value or os.getenv("DULLAHAN_INFERENCE_PROVIDER", "http")
    if provider not in {"http", "openai"}:
        raise ValueError("inference provider must be 'http' or 'openai'")
    return provider


def inference_base_url(provider: InferenceProvider, specific_name: str | None = None) -> str:
    specific = os.getenv(specific_name) if specific_name else None
    common = os.getenv("DULLAHAN_INFERENCE_BASE_URL")
    if specific or common:
        return specific or common or LOCAL_INFERENCE_BASE_URL
    if provider == "openai":
        return os.getenv("OPENAI_BASE_URL", OPENAI_INFERENCE_BASE_URL)
    return os.getenv("DULLAHAN_LOCAL_INFERENCE_BASE_URL", LOCAL_INFERENCE_BASE_URL)


def inference_api_key(provider: InferenceProvider, specific_name: str | None = None) -> str | None:
    specific = os.getenv(specific_name) if specific_name else None
    key = specific or os.getenv("DULLAHAN_INFERENCE_API_KEY")
    if provider == "openai":
        key = key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY or DULLAHAN_INFERENCE_API_KEY is required when "
                "DULLAHAN_INFERENCE_PROVIDER=openai"
            )
    return key


def generation_model(
    provider: InferenceProvider,
    *,
    specific_name: str | None = None,
    local_default: str,
) -> str:
    specific = os.getenv(specific_name) if specific_name else None
    if specific:
        return specific
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    return local_default


def embedding_model(provider: InferenceProvider) -> str:
    default = (
        DEFAULT_OPENAI_EMBEDDING_MODEL
        if provider == "openai"
        else DEFAULT_LOCAL_EMBEDDING_MODEL
    )
    return os.getenv("DULLAHAN_EMBEDDING_MODEL", default)


def tokenizer_base_url(
    provider: InferenceProvider,
    *,
    selected_inference_base_url: str,
) -> str:
    configured = os.getenv("DULLAHAN_TOKENIZER_BASE_URL")
    if configured:
        return configured
    if provider == "http":
        return selected_inference_base_url
    return os.getenv("DULLAHAN_LOCAL_INFERENCE_BASE_URL", LOCAL_INFERENCE_BASE_URL)


def tokenizer_api_key(
    provider: InferenceProvider,
    *,
    selected_inference_api_key: str | None,
) -> str | None:
    configured = os.getenv("DULLAHAN_TOKENIZER_API_KEY")
    if configured:
        return configured
    return selected_inference_api_key if provider == "http" else None
