from __future__ import annotations

import json

from dullahan_inference.config import InferenceConfig
from dullahan_inference.device import DeviceInventory
from dullahan_inference.ollama import OllamaClient
from dullahan_inference.plan import resolve_inference_plan


class FakeResponse:
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {"response": "local answer", "eval_count": 4, "prompt_eval_count": 3}
        ).encode()


def test_ollama_client_uses_non_streaming_generate_api(monkeypatch) -> None:
    config = InferenceConfig(provider="ollama")
    plan = resolve_inference_plan(
        config,
        inventory=DeviceInventory(device="cpu", detection_source="test"),
    )
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("dullahan_inference.ollama.urlopen", fake_urlopen)

    result = OllamaClient(config, plan).complete(
        prompt="question",
        max_tokens=64,
        temperature=0.2,
    )

    assert captured["url"].endswith("/api/generate")
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"] == {
        "num_predict": 64,
        "temperature": 0.2,
        "num_gpu": 0,
    }
    assert result.text == "local answer"
    assert result.completion_tokens == 4
