from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

InferenceProvider = Literal["http", "openai"]


class InferenceHttpError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextGenerationResult:
    text: str
    input_tokens: int | None
    output_tokens: int | None


class OpenAICompatibleTextClient:
    """Calls either Dullahan-compatible completions or OpenAI Responses."""

    def __init__(
        self,
        *,
        base_url: str,
        api_mode: Literal["completions", "responses"] = "completions",
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        opener=urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_mode = api_mode
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    def generate(self, *, model: str, prompt: str, max_tokens: int) -> TextGenerationResult:
        if self.api_mode == "responses":
            endpoint = f"{self.base_url}/responses"
            payload = {"model": model, "input": prompt, "max_output_tokens": max_tokens}
        else:
            endpoint = f"{self.base_url}/completions"
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
            }

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise InferenceHttpError(
                f"inference provider failed with HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise InferenceHttpError(
                f"inference provider request failed: {exc.reason}"
            ) from exc
        except (json.JSONDecodeError, TypeError) as exc:
            raise InferenceHttpError("inference provider returned invalid JSON") from exc

        if not isinstance(data, dict):
            raise InferenceHttpError("inference provider returned a non-object response")
        usage = data.get("usage") or {}
        if self.api_mode == "responses":
            text = self._responses_text(data)
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
        else:
            text = self._completions_text(data)
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
        return TextGenerationResult(
            text=text.strip(),
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        )

    @staticmethod
    def _completions_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        if "text" in first:
            return str(first["text"])
        message = first.get("message")
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return ""

    @staticmethod
    def _responses_text(data: dict[str, Any]) -> str:
        output_text = data.get("output_text")
        if isinstance(output_text, str):
            return output_text
        parts: list[str] = []
        for output in data.get("output") or []:
            if not isinstance(output, dict):
                continue
            for content in output.get("content") or []:
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "\n".join(parts)


def provider_api_mode(provider: InferenceProvider) -> Literal["completions", "responses"]:
    return "responses" if provider == "openai" else "completions"
