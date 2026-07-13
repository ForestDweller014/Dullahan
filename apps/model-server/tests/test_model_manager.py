from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from server import app as model_server


def model_directory(path: Path, *, quantization: str | None = None) -> Path:
    path.mkdir()
    config = {}
    if quantization:
        config["quantization_config"] = {"quant_method": quantization}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def add_adapter(path: Path, name: str = "legal", base_model: str = "Qwen/Qwen3-8B") -> Path:
    adapter = path / "adapters" / name
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    return adapter


def archive_bytes(path: Path) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as stream:
        stream.add(path, arcname=path.name)
    return payload.getvalue()


# Verifies that the model manager exposes exactly the supported CPU and CUDA variants.
def test_manager_variants_are_exactly_cpu_and_cuda() -> None:
    assert {backend.value for backend in model_server.Backend} == {"cpu", "cuda"}


# Verifies that GPTQ and AWQ packages default to CUDA-only backend support.
@pytest.mark.parametrize("quantization", ["gptq", "awq"])
def test_cuda_quantization_defaults_to_cuda_only(
    tmp_path: Path,
    quantization: str,
) -> None:
    model = model_directory(tmp_path / "model", quantization=quantization)

    manifest = model_server.prepare_manifest(model, name="expert", source="test")

    assert manifest["quantization"] == quantization
    assert manifest["supported_backends"] == ["cuda"]


# Verifies that GGUF packages default to both CPU and CUDA backend support.
def test_gguf_defaults_to_cpu_and_cuda(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    gguf = model / "expert.gguf"
    gguf.write_bytes(b"weights")

    manifest = model_server.prepare_manifest(model, name="expert", source="test")

    assert manifest["quantization"] == "gguf"
    assert manifest["supported_backends"] == ["cpu", "cuda"]
    assert model_server.launch_model_path(model, manifest) == gguf


# Verifies that CPU rejects CUDA only checkpoint.
def test_cpu_rejects_cuda_only_checkpoint() -> None:
    with pytest.raises(HTTPException, match="does not support the cpu backend"):
        model_server.validate_backend_compatibility(
            {"supported_backends": ["cuda"]},
            model_server.Backend.CPU,
        )


# Verifies that archive traversal is rejected.
def test_archive_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo("../escape")
        payload = b"bad"
        info.size = len(payload)
        stream.addfile(info, io.BytesIO(payload))

    with pytest.raises(HTTPException, match="Unsafe archive path"):
        model_server.safe_extract_tar(archive, tmp_path / "destination")


# Verifies that activation cannot override manager flags.
@pytest.mark.parametrize(
    "args",
    [["--model", "/tmp/other"], ["--port=9999"], ["--device", "cpu"]],
)
def test_activation_cannot_override_manager_flags(args: list[str]) -> None:
    with pytest.raises(HTTPException, match="controlled by the model server"):
        model_server.validate_activation_args(args)


# Verifies that LoRA only package uses named base and stored adapters.
def test_lora_only_package_uses_named_base_and_stored_adapters(tmp_path: Path) -> None:
    package = tmp_path / "expert"
    package.mkdir()
    adapter = add_adapter(package)
    manifest = model_server.prepare_manifest(
        package,
        name="expert",
        source="test",
        package_mode="lora_only",
        base_model="Qwen/Qwen3-8B",
    )

    model_server.validate_model_package(package)

    assert model_server.launch_model_path(package, manifest) == "Qwen/Qwen3-8B"
    assert model_server.lora_activation_args(
        package,
        manifest,
        max_loras=4,
        max_cpu_loras=8,
    ) == [
        "--enable-lora",
        "--max-loras",
        "4",
        "--max-cpu-loras",
        "8",
        "--lora-modules",
        f"legal={adapter.resolve()}",
    ]


# Verifies that activation command loads named base and LoRA adapters.
def test_activation_command_loads_named_base_and_lora_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "models"
    package = storage / "expert"
    package.mkdir(parents=True)
    adapter = add_adapter(package)
    model_server.prepare_manifest(
        package,
        name="expert",
        source="test",
        package_mode="lora_only",
        base_model="Qwen/Qwen3-8B",
    )
    monkeypatch.setattr(model_server, "MODEL_ROOT", storage)
    monkeypatch.setattr(model_server, "ADMIN_TOKEN", "a" * 32)
    monkeypatch.setattr(model_server, "EXTRA_ARGS", [])
    captured = {}

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **_kwargs):
            captured["command"] = command
            self.running = True

        def poll(self):
            return None if self.running else self.returncode

        def send_signal(self, _signal):
            self.running = False

        def wait(self):
            return self.returncode

        def kill(self):
            self.running = False

    async def ready(_process):
        return None

    monkeypatch.setattr(model_server.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(model_server, "wait_ready", ready)

    with TestClient(model_server.app) as client:
        response = client.post(
            "/admin/models/expert/activate",
            headers={"X-Admin-Token": "a" * 32},
            json={"extra_args": []},
        )
        assert response.status_code == 200

    command = captured["command"]
    assert command[0:3] == ["vllm", "serve", "Qwen/Qwen3-8B"]
    assert "--enable-lora" in command
    assert command[command.index("--max-loras") + 1] == "4"
    assert command[command.index("--max-cpu-loras") + 1] == "8"
    assert f"legal={adapter.resolve()}" in command


# Verifies that activation rejects invalid LoRA capacity.
def test_activation_rejects_invalid_lora_capacity() -> None:
    with pytest.raises(ValueError, match="max_cpu_loras"):
        model_server.ActivateRequest(max_loras=8, max_cpu_loras=4)


# Verifies that model crud metadata and LoRA only export round trip.
def test_model_crud_metadata_and_lora_only_export_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "models"
    storage.mkdir()
    monkeypatch.setattr(model_server, "MODEL_ROOT", storage)
    monkeypatch.setattr(model_server, "ADMIN_TOKEN", "a" * 32)
    monkeypatch.setattr(model_server, "_active_model", None)

    source = model_directory(tmp_path / "source")
    add_adapter(source)
    model_server.prepare_manifest(
        source,
        name="expert",
        source="test",
        base_model="Qwen/Qwen3-8B",
    )
    headers = {"X-Admin-Token": "a" * 32}

    with TestClient(model_server.app) as client:
        created = client.put(
            "/admin/models/expert/archive",
            headers=headers,
            files={"file": ("expert.tar.gz", archive_bytes(source), "application/gzip")},
        )
        assert created.status_code == 201

        metadata = client.get("/admin/models/expert", headers=headers)
        assert metadata.status_code == 200
        body = metadata.json()
        assert body["name"] == "expert"
        assert body["manifest"]["base_model"] == "Qwen/Qwen3-8B"
        assert body["adapters"][0]["name"] == "legal"

        updated = client.put(
            "/admin/models/expert/archive?replace=true",
            headers=headers,
            files={"file": ("expert.tar.gz", archive_bytes(source), "application/gzip")},
        )
        assert updated.status_code == 201

        full_export = client.get("/admin/models/expert/archive?mode=full", headers=headers)
        assert full_export.status_code == 200
        full_archive = tmp_path / "full.tar.gz"
        full_archive.write_bytes(full_export.content)
        with tarfile.open(full_archive, "r:gz") as stream:
            full_names = set(stream.getnames())
        assert "expert/model.safetensors" in full_names
        assert "expert/adapters/legal/adapter_model.safetensors" in full_names

        exported = client.get(
            "/admin/models/expert/archive?mode=lora_only",
            headers=headers,
        )
        assert exported.status_code == 200
        thin_archive = tmp_path / "thin.tar.gz"
        thin_archive.write_bytes(exported.content)
        with tarfile.open(thin_archive, "r:gz") as stream:
            names = set(stream.getnames())
            manifest_member = stream.extractfile("expert/dullahan-model.json")
            assert manifest_member is not None
            thin_manifest = json.loads(manifest_member.read())
        assert "expert/config.json" not in names
        assert "expert/model.safetensors" not in names
        assert "expert/adapters/legal/adapter_config.json" in names
        assert "expert/adapters/legal/adapter_model.safetensors" in names
        assert thin_manifest["package_mode"] == "lora_only"
        assert thin_manifest["base_model"] == "Qwen/Qwen3-8B"

        deleted = client.delete("/admin/models/expert", headers=headers)
        assert deleted.status_code == 200

        restored = client.put(
            "/admin/models/expert/archive",
            headers=headers,
            files={"file": ("expert-lora-only.tar.gz", exported.content, "application/gzip")},
        )
        assert restored.status_code == 201
        restored_metadata = client.get("/admin/models/expert", headers=headers).json()
        assert restored_metadata["manifest"]["package_mode"] == "lora_only"
        assert restored_metadata["manifest"]["base_model"] == "Qwen/Qwen3-8B"
        assert [adapter["name"] for adapter in restored_metadata["adapters"]] == ["legal"]


# Verifies that get model metadata requires auth and returns 404.
def test_get_model_metadata_requires_auth_and_returns_404(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "models"
    storage.mkdir()
    monkeypatch.setattr(model_server, "MODEL_ROOT", storage)
    monkeypatch.setattr(model_server, "ADMIN_TOKEN", "a" * 32)

    with TestClient(model_server.app) as client:
        assert client.get("/admin/models/missing").status_code == 401
        response = client.get(
            "/admin/models/missing",
            headers={"X-Admin-Token": "a" * 32},
        )
        assert response.status_code == 404
