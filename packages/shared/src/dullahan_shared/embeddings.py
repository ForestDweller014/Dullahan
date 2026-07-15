from __future__ import annotations

import json
import math
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingError(RuntimeError):
    pass


class EmbeddingModel(Protocol):
    model_id: str
    dimensions: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbeddingModel:
    """Semantic embedding client for the Dullahan inference boundary."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimensions: int,
        timeout_seconds: float = 120.0,
        api_key: str | None = None,
        request_dimensions: bool = False,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.base_url = base_url.rstrip("/")
        self.model_id = model
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.request_dimensions = request_dimensions

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, object] = {"model": self.model_id, "input": texts}
        if self.request_dimensions:
            payload["dimensions"] = self.dimensions
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingError(
                f"embedding provider failed with HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise EmbeddingError(f"embedding provider request failed: {exc.reason}") from exc

        rows = sorted(payload.get("data", []), key=lambda item: int(item.get("index", 0)))
        embeddings = [list(map(float, row.get("embedding", []))) for row in rows]
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"embedding provider returned {len(embeddings)} vectors for {len(texts)} inputs"
            )
        if any(len(vector) != self.dimensions for vector in embeddings):
            actual = sorted({len(vector) for vector in embeddings})
            raise EmbeddingError(
                f"embedding dimensions {actual} do not match configured {self.dimensions}"
            )
        return embeddings


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    score = sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return max(-1.0, min(1.0, score))
