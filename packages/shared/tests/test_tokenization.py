from __future__ import annotations

import json

import dullahan_shared.tokenization as tokenization_module
import pytest
from dullahan_shared.tokenization import InferenceTokenCounter, TokenizationError


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


# Verifies native token-count HTTP parsing while the external inference server is mocked.
def test_inference_token_counter_uses_tokenize_endpoint_and_caches(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.full_url, json.loads(request.data), timeout))
        return FakeResponse({"count": 7, "model": "generation-model"})

    monkeypatch.setattr(tokenization_module, "urlopen", fake_urlopen)
    counter = InferenceTokenCounter(
        base_url="http://inference.local/v1",
        model="generation-model",
        timeout_seconds=3,
    )

    assert counter.count("tokenize this") == 7
    assert counter.count("tokenize this") == 7
    assert requests == [
        (
            "http://inference.local/tokenize",
            {"model": "generation-model", "prompt": "tokenize this"},
            3,
        )
    ]


# Verifies invalid native usage rejection while the external inference response is mocked.
def test_inference_token_counter_rejects_missing_native_count(monkeypatch) -> None:
    monkeypatch.setattr(
        tokenization_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse({"model": "generation-model"}),
    )
    counter = InferenceTokenCounter(
        base_url="http://inference.local/v1",
        model="generation-model",
    )

    with pytest.raises(TokenizationError, match="valid count"):
        counter.count("text")
