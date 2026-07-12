from __future__ import annotations

import json
from pathlib import Path

import pytest
from dullahan_inference.config import InferenceConfig
from dullahan_inference.device import DeviceInventory
from dullahan_inference.model_server import (
    ModelServerError,
    activate_model_server,
    export_model_server_package,
    get_model_server_metadata,
)
from dullahan_inference.plan import resolve_inference_plan


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self, *_args) -> bytes:
        return b'{"active_model":"expert"}'


def test_container_plan_selects_cpu_endpoint_and_gguf() -> None:
    config = InferenceConfig(
        model_server={"enabled": True, "model": "expert"},
    )

    plan = resolve_inference_plan(
        config,
        inventory=DeviceInventory(device="metal", detection_source="test"),
    )

    assert plan.engine == "vllm-model-server"
    assert plan.container_backend == "cpu"
    assert plan.quantization.value == "gguf"
    assert plan.api_base_url == "http://127.0.0.1:8001/v1"


def test_container_plan_selects_cuda_endpoint_and_gptq() -> None:
    config = InferenceConfig(
        model_server={"enabled": True, "model": "expert"},
    )

    plan = resolve_inference_plan(
        config,
        inventory=DeviceInventory(device="cuda", cuda_device_count=1),
    )

    assert plan.container_backend == "cuda"
    assert plan.quantization.value == "gptq"
    assert plan.api_base_url == "http://127.0.0.1:8002/v1"


def test_model_server_activation_uses_admin_token(monkeypatch) -> None:
    config = InferenceConfig(model_server={"enabled": True, "model": "expert"})
    plan = resolve_inference_plan(
        config,
        inventory=DeviceInventory(device="cpu"),
    )
    captured = {}
    monkeypatch.setenv("MODEL_ADMIN_TOKEN", "secret")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["token"] = request.headers["X-admin-token"]
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("dullahan_inference.model_server.urlopen", fake_urlopen)

    result = activate_model_server(plan)

    assert captured["url"].endswith("/admin/models/expert/activate")
    assert captured["token"] == "secret"
    assert captured["payload"] == {
        "extra_args": [],
        "max_loras": 4,
        "max_cpu_loras": 8,
    }
    assert result == {"active_model": "expert"}


def test_model_server_activation_requires_token(monkeypatch) -> None:
    config = InferenceConfig(model_server={"enabled": True})
    plan = resolve_inference_plan(config, inventory=DeviceInventory(device="cpu"))
    monkeypatch.delenv("MODEL_ADMIN_TOKEN", raising=False)

    with pytest.raises(ModelServerError, match="MODEL_ADMIN_TOKEN"):
        activate_model_server(plan)


def test_model_server_metadata_uses_dedicated_endpoint(monkeypatch) -> None:
    config = InferenceConfig(model_server={"enabled": True, "model": "expert"})
    plan = resolve_inference_plan(config, inventory=DeviceInventory(device="cpu"))
    monkeypatch.setenv("MODEL_ADMIN_TOKEN", "secret")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.method
        return FakeResponse()

    monkeypatch.setattr("dullahan_inference.model_server.urlopen", fake_urlopen)

    assert get_model_server_metadata(plan) == {"active_model": "expert"}
    assert captured == {
        "url": "http://127.0.0.1:8001/admin/models/expert",
        "method": "GET",
    }


def test_lora_only_export_mode_flows_from_config_to_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = InferenceConfig(
        model_server={"enabled": True, "model": "expert", "export_mode": "lora_only"}
    )
    plan = resolve_inference_plan(config, inventory=DeviceInventory(device="cpu"))
    monkeypatch.setenv("MODEL_ADMIN_TOKEN", "secret")
    captured = {}

    class ArchiveResponse(FakeResponse):
        def __init__(self):
            self.remaining = b"package"

        def read(self, *_args) -> bytes:
            payload, self.remaining = self.remaining, b""
            return payload

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return ArchiveResponse()

    monkeypatch.setattr("dullahan_inference.model_server.urlopen", fake_urlopen)
    destination = tmp_path / "expert.tar.gz"

    assert export_model_server_package(plan, destination) == destination
    assert destination.read_bytes() == b"package"
    assert captured["url"].endswith("/admin/models/expert/archive?mode=lora_only")
    assert plan.model_export_mode == "lora_only"


def test_model_server_lora_capacity_flows_to_plan() -> None:
    config = InferenceConfig(
        model_server={"enabled": True, "max_loras": 6, "max_cpu_loras": 12}
    )

    plan = resolve_inference_plan(config, inventory=DeviceInventory(device="cpu"))

    assert plan.max_loras == 6
    assert plan.max_cpu_loras == 12


def test_model_server_rejects_cpu_lora_cache_below_batch_capacity() -> None:
    with pytest.raises(ValueError, match="max_cpu_loras"):
        InferenceConfig(model_server={"max_loras": 8, "max_cpu_loras": 4})
