from __future__ import annotations

from dataclasses import dataclass
from urllib.request import urlopen

from dullahan_shared.inference import InferenceHttpError, OpenAICompatibleTextClient


@dataclass(frozen=True)
class ModelRequest:
    model: str
    prompt: str
    max_tokens: int = 512


@dataclass(frozen=True)
class ModelResult:
    text: str
    provider: str
    token_count: int


class ModelProvider:
    def complete(self, request: ModelRequest) -> ModelResult:
        raise NotImplementedError


class ModelProviderError(RuntimeError):
    pass


class OpenAICompatibleHttpProvider(ModelProvider):
    """HTTP provider for Dullahan completions or hosted OpenAI Responses."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 30.0,
        api_mode: str = "completions",
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.api_mode = api_mode
        self.api_key = api_key

    def complete(self, request: ModelRequest) -> ModelResult:
        try:
            result = OpenAICompatibleTextClient(
                base_url=self.base_url,
                api_mode=self.api_mode,
                api_key=self.api_key,
                timeout_seconds=self.timeout_seconds,
                opener=urlopen,
            ).generate(
                model=request.model,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
            )
        except InferenceHttpError as exc:
            raise ModelProviderError(f"model provider failed: {exc}") from exc
        if not result.text:
            raise ModelProviderError("model provider response contained no text")
        if result.output_tokens is None or result.output_tokens < 0:
            raise ModelProviderError(
                "model provider response contained no native completion token usage"
            )
        return ModelResult(
            text=result.text,
            provider="openai-compatible-http",
            token_count=result.output_tokens,
        )
