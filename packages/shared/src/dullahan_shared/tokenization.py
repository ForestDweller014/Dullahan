from __future__ import annotations

import json
from threading import Lock
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class TokenizationError(RuntimeError):
    pass


class TokenCounter(Protocol):
    model_id: str

    def count(self, text: str) -> int: ...


class InferenceTokenCounter:
    """Counts tokens through the serving model's native tokenizer usage."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        normalized = base_url.rstrip("/")
        self.endpoint = f"{normalized.removesuffix('/v1')}/tokenize"
        self.model_id = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self._cache: dict[str, int] = {}
        self._cache_lock = Lock()

    def count(self, text: str) -> int:
        with self._cache_lock:
            cached = self._cache.get(text)
        if cached is not None:
            return cached
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.endpoint,
            data=json.dumps({"model": self.model_id, "prompt": text}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TokenizationError(
                f"tokenizer provider failed with HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise TokenizationError(f"tokenizer provider request failed: {exc.reason}") from exc
        count = payload.get("count")
        if not isinstance(count, int) or count < 0:
            raise TokenizationError("tokenizer provider response contained no valid count")
        with self._cache_lock:
            if len(self._cache) >= 4096:
                self._cache.pop(next(iter(self._cache)))
            self._cache[text] = count
        return count
