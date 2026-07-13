from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen


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
    """HTTP provider for OpenAI-compatible completion APIs, including SGLang."""

    def __init__(self, *, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def complete(self, request: ModelRequest) -> ModelResult:
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
        }
        http_request = UrlRequest(
            url=f"{self.base_url}/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )

        try:
            with urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ModelProviderError(
                f"model provider failed with HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise ModelProviderError(f"model provider request failed: {exc.reason}") from exc

        data = json.loads(response_body)
        text = self._extract_text(data)
        usage = data.get("usage", {})
        token_count = usage.get("completion_tokens")
        if not isinstance(token_count, int) or token_count < 0:
            raise ModelProviderError(
                "model provider response contained no native completion token usage"
            )
        return ModelResult(
            text=text,
            provider="openai-compatible-http",
            token_count=int(token_count),
        )

    def _extract_text(self, data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise ModelProviderError("model provider response contained no choices")

        first_choice = choices[0]
        if "text" in first_choice:
            return str(first_choice["text"]).strip()

        message = first_choice.get("message")
        if isinstance(message, dict) and "content" in message:
            return str(message["content"]).strip()

        raise ModelProviderError("model provider response contained no text")
