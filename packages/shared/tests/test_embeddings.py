from __future__ import annotations

import json

import dullahan_shared.embeddings as embedding_module
import pytest
from dullahan_shared.embeddings import (
    EmbeddingError,
    OpenAICompatibleEmbeddingModel,
    cosine_similarity,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


# Verifies the embedding HTTP contract while the external inference server is mocked.
def test_embedding_model_posts_batch_and_preserves_vector_order(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        )

    monkeypatch.setattr(embedding_module, "urlopen", fake_urlopen)
    model = OpenAICompatibleEmbeddingModel(
        base_url="http://inference.local/v1",
        model="semantic-model",
        dimensions=2,
        timeout_seconds=4,
    )

    vectors = model.embed_many(["first", "second"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert captured["url"] == "http://inference.local/v1/embeddings"
    assert captured["payload"] == {
        "model": "semantic-model",
        "input": ["first", "second"],
    }
    assert captured["timeout"] == 4


# Verifies dimension validation while the external inference response is mocked.
def test_embedding_model_rejects_unexpected_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(
        embedding_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            {"data": [{"index": 0, "embedding": [1.0]}]}
        ),
    )
    model = OpenAICompatibleEmbeddingModel(
        base_url="http://inference.local/v1",
        model="semantic-model",
        dimensions=2,
    )

    with pytest.raises(EmbeddingError, match="dimensions"):
        model.embed("text")


# Verifies cosine similarity normalization without mocking inference functionality.
def test_cosine_similarity_normalizes_vectors_and_rejects_mismatched_dimensions() -> None:
    assert cosine_similarity([2.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="dimensions"):
        cosine_similarity([1.0], [1.0, 0.0])
