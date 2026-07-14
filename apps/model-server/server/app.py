from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field, model_validator

MODEL_ROOT = Path(os.getenv("MODEL_ROOT", "/models")).resolve()
VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
STARTUP_TIMEOUT = int(os.getenv("VLLM_STARTUP_TIMEOUT", "900"))
ADMIN_TOKEN = os.getenv("MODEL_ADMIN_TOKEN", "change-me")
EXTRA_ARGS = json.loads(os.getenv("VLLM_EXTRA_ARGS", "[]"))
BACKEND_NAME = os.getenv("DULLAHAN_BACKEND", "cpu")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_process: subprocess.Popen[bytes] | None = None
_active_model: str | None = None
_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        await stop_vllm()


app = FastAPI(title="Dullahan Model Server", version="0.1.0", lifespan=lifespan)


class Backend(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class ModelQuantization(StrEnum):
    NONE = "none"
    GPTQ = "gptq"
    GGUF = "gguf"
    AWQ = "awq"


class PackageMode(StrEnum):
    LORA_ONLY = "lora_only"


class ExportMode(StrEnum):
    LORA_ONLY = "lora_only"


class HFImport(BaseModel):
    name: str
    repo_id: str
    revision: str | None = None
    hf_token: str | None = Field(default=None, repr=False)
    replace: bool = False
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    quantization: ModelQuantization | None = None
    supported_backends: list[Backend] | None = None
    base_model: str | None = None
    package_mode: PackageMode = PackageMode.LORA_ONLY
    adapter_name: str | None = None


def positive_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise RuntimeError(f"{name} must be at least 1")
    return value


DEFAULT_MAX_LORAS = positive_env("VLLM_MAX_LORAS", 4)
DEFAULT_MAX_CPU_LORAS = positive_env("VLLM_MAX_CPU_LORAS", 8)


class ActivateRequest(BaseModel):
    extra_args: list[str] = Field(default_factory=list)
    max_loras: int = Field(default=DEFAULT_MAX_LORAS, ge=1)
    max_cpu_loras: int = Field(default=DEFAULT_MAX_CPU_LORAS, ge=1)

    @model_validator(mode="after")
    def validate_lora_capacity(self) -> ActivateRequest:
        if self.max_cpu_loras < self.max_loras:
            raise ValueError("max_cpu_loras must be >= max_loras")
        return self


BACKEND = Backend(BACKEND_NAME)
DEFAULT_EXPORT_MODE = ExportMode.LORA_ONLY
ADAPTERS_DIRECTORY = "adapters"
ADAPTER_WEIGHT_PATTERNS = ("adapter_model.safetensors", "adapter_model.bin")
BASE_WEIGHT_PATTERNS = (
    "model.safetensors",
    "model-*.safetensors",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "*.gguf",
    "*.safetensors.index.json",
    "*.bin.index.json",
)
PROTECTED_ACTIVATION_FLAGS = {
    "--device",
    "--host",
    "--model",
    "--max-cpu-loras",
    "--max-loras",
    "--port",
    "--quantization",
    "--served-model-name",
    "--tokenizer",
}


def require_admin(request: Request) -> None:
    if ADMIN_TOKEN == "change-me" or len(ADMIN_TOKEN) < 32:
        raise HTTPException(status_code=503, detail="MODEL_ADMIN_TOKEN is not securely configured")
    supplied = request.headers.get("x-admin-token")
    if supplied != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token")


def checked_name(name: str) -> str:
    if not NAME_RE.fullmatch(name):
        raise HTTPException(status_code=400, detail="Invalid model name")
    return name


def model_dir(name: str) -> Path:
    return MODEL_ROOT / checked_name(name)


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    payload = {**data, "updated_at": int(time.time())}
    (path / "dullahan-model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def detect_quantization(path: Path) -> ModelQuantization:
    if any(path.glob("*.gguf")):
        return ModelQuantization.GGUF
    config_path = path / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quantization = config.get("quantization_config") or {}
        method = str(quantization.get("quant_method", "")).lower()
        if "gptq" in method:
            return ModelQuantization.GPTQ
        if "awq" in method:
            return ModelQuantization.AWQ
    return ModelQuantization.NONE


def default_supported_backends(
    quantization: ModelQuantization,
) -> list[Backend]:
    if quantization in {ModelQuantization.GPTQ, ModelQuantization.AWQ}:
        return [Backend.CUDA]
    return [Backend.CPU, Backend.CUDA]


def model_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "dullahan-model.json"
    return json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}


def lora_adapters(path: Path) -> list[dict[str, Any]]:
    adapters_root = path / ADAPTERS_DIRECTORY
    if not adapters_root.is_dir():
        return []
    result = []
    for adapter in sorted(adapters_root.iterdir()):
        if not adapter.is_dir() or adapter.name.startswith("."):
            continue
        result.append(lora_adapter_record(path, adapter))
    return result


def lora_adapter_record(package: Path, adapter: Path) -> dict[str, Any]:
    name = checked_name(adapter.name)
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"LoRA adapter {name!r} is missing adapter_config.json",
        )
    weights = [
        weight
        for pattern in ADAPTER_WEIGHT_PATTERNS
        for weight in adapter.glob(pattern)
        if weight.is_file()
    ]
    if not weights:
        raise HTTPException(
            status_code=400,
            detail=(
                f"LoRA adapter {name!r} must contain adapter_model.safetensors "
                "or adapter_model.bin"
            ),
        )
    base_weights = sorted(
        weight.relative_to(adapter).as_posix()
        for pattern in BASE_WEIGHT_PATTERNS
        for weight in adapter.rglob(pattern)
        if weight.is_file()
    )
    if base_weights:
        raise HTTPException(
            status_code=400,
            detail=(
                f"LoRA adapter {name!r} contains base-model weight files: "
                + ", ".join(base_weights)
            ),
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "directory": f"{ADAPTERS_DIRECTORY}/{name}",
        "storage_path": str(adapter.resolve()),
        "base_model": config.get("base_model_name_or_path"),
        "bytes": dir_size(adapter),
    }


def package_mode(path: Path, manifest: dict[str, Any] | None = None) -> PackageMode:
    manifest = manifest if manifest is not None else model_manifest(path)
    try:
        return PackageMode(manifest.get("package_mode", PackageMode.LORA_ONLY))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid model package_mode") from exc


def prepare_manifest(
    path: Path,
    *,
    name: str,
    source: str,
    quantization: ModelQuantization | str | None = None,
    supported_backends: list[Backend | str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    existing = model_manifest(path)
    adapters = lora_adapters(path)
    adapter_base_models = {adapter["base_model"] for adapter in adapters if adapter["base_model"]}
    inferred_base_model = next(iter(adapter_base_models)) if len(adapter_base_models) == 1 else None
    quantization_mode = ModelQuantization(
        quantization or existing.get("quantization") or detect_quantization(path)
    )
    backends = [
        Backend(value)
        for value in (
            supported_backends
            or existing.get("supported_backends")
            or default_supported_backends(quantization_mode)
        )
    ]
    payload = {
        **existing,
        **extra,
        "name": name,
        "source": source,
        "format": "huggingface",
        "package_mode": package_mode(path, {**existing, **extra}).value,
        "quantization": quantization_mode.value,
        "supported_backends": [backend.value for backend in backends],
        "base_model": extra.get("base_model") or existing.get("base_model") or inferred_base_model,
        "adapters": [
            {key: value for key, value in adapter.items() if key != "storage_path"}
            for adapter in adapters
        ],
    }
    write_manifest(path, payload)
    return payload


def validate_activation_args(args: list[str]) -> None:
    for arg in args:
        flag = arg.split("=", 1)[0]
        if flag in PROTECTED_ACTIVATION_FLAGS:
            raise HTTPException(
                status_code=400,
                detail=f"Activation argument {flag} is controlled by the model server",
            )


def launch_model_path(path: Path, manifest: dict[str, Any]) -> Path | str:
    package_mode(path, manifest)
    base_model = manifest.get("base_model")
    if not base_model:
        raise HTTPException(status_code=400, detail="LoRA package has no base_model")
    return str(base_model)


def lora_activation_args(
    path: Path,
    manifest: dict[str, Any],
    *,
    max_loras: int,
    max_cpu_loras: int,
) -> list[str]:
    adapters = manifest.get("adapters") or lora_adapters(path)
    if not adapters:
        return []
    modules = []
    for adapter in adapters:
        adapter_path = (path / adapter["directory"]).resolve()
        if path.resolve() not in adapter_path.parents or not adapter_path.is_dir():
            raise HTTPException(status_code=400, detail="LoRA adapter path is invalid")
        modules.append(f"{checked_name(adapter['name'])}={adapter_path}")
    return [
        "--enable-lora",
        "--max-loras",
        str(max_loras),
        "--max-cpu-loras",
        str(max_cpu_loras),
        "--lora-modules",
        *modules,
    ]


def validate_backend_compatibility(
    manifest: dict[str, Any],
    backend: Backend = BACKEND,
) -> None:
    try:
        supported = [Backend(value) for value in manifest.get("supported_backends", [])]
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Model manifest contains an invalid supported_backends value",
        ) from exc
    if supported and backend not in supported:
        raise HTTPException(
            status_code=409,
            detail=f"Model does not support the {backend.value} backend",
        )


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def validate_adapter_only_layout(path: Path) -> None:
    allowed = {"dullahan-model.json", ADAPTERS_DIRECTORY}
    unexpected = sorted(item.name for item in path.iterdir() if item.name not in allowed)
    if unexpected:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model packages are adapter-only; unexpected top-level files: "
                + ", ".join(unexpected)
            ),
        )
    adapters_root = path / ADAPTERS_DIRECTORY
    if adapters_root.is_dir():
        invalid_entries = sorted(
            item.name
            for item in adapters_root.iterdir()
            if not item.is_dir() or item.name.startswith(".")
        )
        if invalid_entries:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The adapters directory may contain only named adapter directories: "
                    + ", ".join(invalid_entries)
                ),
            )


def validate_model_package(path: Path) -> dict[str, Any]:
    manifest = model_manifest(path)
    mode = package_mode(path, manifest)
    validate_adapter_only_layout(path)
    adapters = lora_adapters(path)
    base_model = manifest.get("base_model")
    if not base_model:
        raise HTTPException(
            status_code=400,
            detail="LoRA packages must declare base_model in dullahan-model.json",
        )
    if not adapters:
        raise HTTPException(
            status_code=400,
            detail="LoRA packages must contain at least one LoRA adapter",
        )
    mismatched = sorted(
        adapter["name"]
        for adapter in adapters
        if adapter["base_model"] and adapter["base_model"] != base_model
    )
    if mismatched:
        raise HTTPException(
            status_code=409,
            detail=(
                f"LoRA adapters do not target declared base_model {base_model!r}: "
                + ", ".join(mismatched)
            ),
        )
    return {**manifest, "package_mode": mode.value, "adapters": adapters}


def model_record(path: Path) -> dict[str, Any]:
    manifest = model_manifest(path)
    adapters = lora_adapters(path)
    if manifest and manifest.get("adapters") != adapters:
        manifest = {**manifest, "adapters": adapters}
    return {
        "name": path.name,
        "active": path.name == _active_model,
        "storage_directory": str(path.resolve()),
        "bytes": dir_size(path),
        "manifest": manifest,
        "adapters": adapters,
    }


def safe_extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as tf:
        base = destination.resolve()
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if target != base and base not in target.parents:
                raise HTTPException(status_code=400, detail="Unsafe archive path")
            if member.issym() or member.islnk():
                raise HTTPException(
                    status_code=400, detail="Links are not allowed in model archives"
                )
            if not (member.isfile() or member.isdir()):
                raise HTTPException(
                    status_code=400,
                    detail="Special files are not allowed in model archives",
                )
        tf.extractall(destination, filter="data")


def normalize_single_root(path: Path) -> Path:
    children = [p for p in path.iterdir() if p.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir() and not (path / "config.json").exists():
        return children[0]
    return path


def replace_model_directory(source: Path, target: Path) -> None:
    staging = MODEL_ROOT / f".{target.name}.staging"
    backup = MODEL_ROOT / f".{target.name}.old"
    if staging.exists():
        shutil.rmtree(staging)
    if backup.exists():
        shutil.rmtree(backup)
    source.rename(staging)
    try:
        if target.exists():
            target.rename(backup)
        staging.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


async def stop_vllm() -> None:
    global _process, _active_model
    if _process is None:
        _active_model = None
        return
    if _process.poll() is None:
        _process.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.to_thread(_process.wait), timeout=30)
        except TimeoutError:
            _process.kill()
            await asyncio.to_thread(_process.wait)
    _process = None
    _active_model = None


async def wait_ready(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/health"
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited with status {process.returncode}")
            try:
                response = await client.get(url)
                if response.status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise RuntimeError("Timed out waiting for vLLM to become ready")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "manager": "ok",
        "backend": BACKEND.value,
        "active_model": _active_model,
        "vllm_running": bool(_process and _process.poll() is None),
    }


@app.get("/admin/models")
async def list_models(request: Request) -> list[dict[str, Any]]:
    require_admin(request)
    result = []
    for path in sorted(MODEL_ROOT.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        result.append(model_record(path))
    return result


@app.get("/admin/models/{name}")
async def get_model(name: str, request: Request) -> dict[str, Any]:
    require_admin(request)
    target = model_dir(name)
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="Model not found")
    return model_record(target)


@app.post("/admin/models/hf", status_code=201)
async def import_hf(spec: HFImport, request: Request) -> dict[str, Any]:
    require_admin(request)
    name = checked_name(spec.name)
    target = model_dir(name)
    if target.exists() and not spec.replace:
        raise HTTPException(status_code=409, detail="Model already exists")
    if name == _active_model:
        raise HTTPException(status_code=409, detail="Cannot replace the active model")

    temp = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=MODEL_ROOT))
    try:
        await asyncio.to_thread(
            snapshot_download,
            repo_id=spec.repo_id,
            revision=spec.revision,
            token=spec.hf_token or os.getenv("HF_TOKEN"),
            local_dir=temp,
            allow_patterns=spec.allow_patterns,
            ignore_patterns=spec.ignore_patterns,
        )
        package = Path(tempfile.mkdtemp(prefix=f".{name}-package-", dir=MODEL_ROOT))
        adapter = package / ADAPTERS_DIRECTORY / checked_name(spec.adapter_name or name)
        adapter.parent.mkdir()
        temp.rename(adapter)
        temp = package
        adapter_metadata = lora_adapter_record(package, adapter)
        base_model = spec.base_model or adapter_metadata["base_model"]
        if not base_model:
            raise HTTPException(
                status_code=400,
                detail=(
                    "LoRA Hugging Face imports require base_model or an "
                    "adapter_config.json that declares base_model_name_or_path"
                ),
            )
        prepare_manifest(
            temp,
            name=name,
            source="huggingface",
            repo_id=spec.repo_id,
            revision=spec.revision,
            base_model=base_model,
            quantization=spec.quantization,
            supported_backends=spec.supported_backends,
            package_mode=PackageMode.LORA_ONLY.value,
        )
        validate_model_package(temp)
        replace_model_directory(temp, target)
        return model_record(target)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        raise


@app.put("/admin/models/{name}/archive", status_code=201)
async def upload_model(
    name: str,
    request: Request,
    file: Annotated[UploadFile, File()],
    replace: bool = False,
) -> dict[str, Any]:
    require_admin(request)
    name = checked_name(name)
    target = model_dir(name)
    if target.exists() and not replace:
        raise HTTPException(status_code=409, detail="Model already exists")
    if name == _active_model:
        raise HTTPException(status_code=409, detail="Cannot replace the active model")

    work = Path(tempfile.mkdtemp(prefix=f".{name}-upload-", dir=MODEL_ROOT))
    archive = work / "model.tar"
    extracted = work / "extracted"
    extracted.mkdir()
    digest = hashlib.sha256()
    try:
        with archive.open("wb") as out:
            while chunk := await file.read(8 * 1024 * 1024):
                digest.update(chunk)
                out.write(chunk)
        safe_extract_tar(archive, extracted)
        source = normalize_single_root(extracted)
        validate_model_package(source)
        prepare_manifest(
            source,
            name=name,
            source="upload",
            sha256=digest.hexdigest(),
            package_mode=PackageMode.LORA_ONLY.value,
        )
        validate_model_package(source)
        replace_model_directory(source, target)
        return {
            **model_record(target),
            "sha256": digest.hexdigest(),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.get("/admin/models/{name}/adapters/{adapter_name}")
async def get_adapter(name: str, adapter_name: str, request: Request) -> dict[str, Any]:
    require_admin(request)
    target = model_dir(name)
    adapter_name = checked_name(adapter_name)
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="Model not found")
    adapter = target / ADAPTERS_DIRECTORY / adapter_name
    if not adapter.is_dir():
        raise HTTPException(status_code=404, detail="LoRA adapter not found")
    return lora_adapter_record(target, adapter)


@app.put("/admin/models/{name}/adapters/{adapter_name}/archive", status_code=201)
async def upload_adapter(
    name: str,
    adapter_name: str,
    request: Request,
    file: Annotated[UploadFile, File()],
    base_model: str | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    require_admin(request)
    name = checked_name(name)
    adapter_name = checked_name(adapter_name)
    target = model_dir(name)
    if name == _active_model:
        raise HTTPException(status_code=409, detail="Cannot modify the active model")

    work = Path(tempfile.mkdtemp(prefix=f".{name}-{adapter_name}-upload-", dir=MODEL_ROOT))
    archive = work / "adapter.tar"
    extracted = work / "extracted"
    extracted.mkdir()
    digest = hashlib.sha256()
    try:
        with archive.open("wb") as out:
            while chunk := await file.read(8 * 1024 * 1024):
                digest.update(chunk)
                out.write(chunk)
        safe_extract_tar(archive, extracted)
        source = normalize_single_root(extracted)

        package = work / "package"
        if target.exists():
            shutil.copytree(target, package)
        else:
            package.mkdir()
            (package / ADAPTERS_DIRECTORY).mkdir()

        manifest = model_manifest(package)
        existing_base_model = manifest.get("base_model")
        if base_model and existing_base_model and base_model != existing_base_model:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Package base_model is {existing_base_model!r}; "
                    f"cannot replace it with {base_model!r}"
                ),
            )

        destination = package / ADAPTERS_DIRECTORY / adapter_name
        if destination.exists() and not replace:
            raise HTTPException(status_code=409, detail="LoRA adapter already exists")
        if destination.exists():
            shutil.rmtree(destination)
        source.rename(destination)
        adapter_metadata = lora_adapter_record(package, destination)
        resolved_base_model = base_model or existing_base_model or adapter_metadata["base_model"]
        if not resolved_base_model:
            raise HTTPException(
                status_code=400,
                detail=(
                    "A new package requires base_model or an adapter_config.json "
                    "that declares base_model_name_or_path"
                ),
            )
        prepare_manifest(
            package,
            name=name,
            source="adapter_upload",
            base_model=resolved_base_model,
            package_mode=PackageMode.LORA_ONLY.value,
        )
        validate_model_package(package)
        replace_model_directory(package, target)
        stored_adapter = target / ADAPTERS_DIRECTORY / adapter_name
        return {
            "name": name,
            "adapter": lora_adapter_record(target, stored_adapter),
            "sha256": digest.hexdigest(),
            "storage_directory": str(target.resolve()),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.delete("/admin/models/{name}/adapters/{adapter_name}")
async def delete_adapter(name: str, adapter_name: str, request: Request) -> dict[str, str]:
    require_admin(request)
    name = checked_name(name)
    adapter_name = checked_name(adapter_name)
    target = model_dir(name)
    if name == _active_model:
        raise HTTPException(status_code=409, detail="Cannot modify the active model")
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="Model not found")
    adapter = target / ADAPTERS_DIRECTORY / adapter_name
    if not adapter.is_dir():
        raise HTTPException(status_code=404, detail="LoRA adapter not found")
    if len(lora_adapters(target)) == 1:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete the package's only adapter; delete the model package instead",
        )

    work = Path(tempfile.mkdtemp(prefix=f".{name}-{adapter_name}-delete-", dir=MODEL_ROOT))
    try:
        package = work / "package"
        shutil.copytree(target, package)
        shutil.rmtree(package / ADAPTERS_DIRECTORY / adapter_name)
        prepare_manifest(
            package,
            name=name,
            source="adapter_delete",
            package_mode=PackageMode.LORA_ONLY.value,
        )
        validate_model_package(package)
        replace_model_directory(package, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"deleted": adapter_name, "model": name}


@app.delete("/admin/models/{name}")
async def delete_model(name: str, request: Request) -> dict[str, str]:
    require_admin(request)
    name = checked_name(name)
    if name == _active_model:
        raise HTTPException(
            status_code=409, detail="Deactivate or activate another model before deleting this one"
        )
    target = model_dir(name)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    shutil.rmtree(target)
    return {"deleted": name}


@app.get("/admin/models/{name}/archive")
async def export_model(name: str, request: Request) -> FileResponse:
    require_admin(request)
    name = checked_name(name)
    source = model_dir(name)
    if not source.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    requested_mode = request.query_params.get("mode")
    if requested_mode not in {None, DEFAULT_EXPORT_MODE.value}:
        raise HTTPException(status_code=400, detail="mode must be lora_only")

    export_dir = MODEL_ROOT / ".exports"
    export_dir.mkdir(exist_ok=True)
    output = export_dir / f"{name}-lora-only.tar.gz"
    temp_output = export_dir / f".{name}-lora-only.tar.gz.tmp"
    if temp_output.exists():
        temp_output.unlink()
    validate_model_package(source)
    with tarfile.open(temp_output, "w:gz") as tf:
        tf.add(source, arcname=name, recursive=True)
    temp_output.replace(output)
    return FileResponse(output, media_type="application/gzip", filename=output.name)


@app.post("/admin/models/{name}/activate")
async def activate_model(name: str, spec: ActivateRequest, request: Request) -> dict[str, Any]:
    require_admin(request)
    global _process, _active_model
    name = checked_name(name)
    path = model_dir(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    validate_model_package(path)
    manifest = model_manifest(path)
    validate_backend_compatibility(manifest)
    validate_activation_args(spec.extra_args)
    launch_path = launch_model_path(path, manifest)

    async with _lock:
        await stop_vllm()
        command = [
            "vllm",
            "serve",
            str(launch_path),
            "--served-model-name",
            name,
            "--host",
            VLLM_HOST,
            "--port",
            str(VLLM_PORT),
            *EXTRA_ARGS,
            *lora_activation_args(
                path,
                manifest,
                max_loras=spec.max_loras,
                max_cpu_loras=spec.max_cpu_loras,
            ),
            *spec.extra_args,
        ]
        if manifest.get("quantization") == ModelQuantization.GGUF:
            command.extend(["--quantization", "gguf"])
            if manifest.get("base_model"):
                command.extend(["--tokenizer", str(manifest["base_model"])])
        log_path = MODEL_ROOT / "vllm.log"
        log = log_path.open("ab", buffering=0)
        _process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            await wait_ready(_process)
        except Exception as exc:
            await stop_vllm()
            raise HTTPException(
                status_code=500, detail=f"vLLM failed to start: {exc}. See {log_path}"
            ) from exc
        _active_model = name
        return {"active_model": name, "command": command}


@app.post("/admin/deactivate")
async def deactivate(request: Request) -> dict[str, str]:
    require_admin(request)
    async with _lock:
        await stop_vllm()
    return {"status": "stopped"}


async def proxy_stream(response: httpx.Response, client: httpx.AsyncClient) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        await response.aclose()
        await client.aclose()


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_vllm(path: str, request: Request):
    if not _process or _process.poll() is not None:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "No model is active", "type": "server_unavailable"}},
        )
    client = httpx.AsyncClient(timeout=None)
    upstream_url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/{path}"
    upstream_request = client.build_request(
        request.method,
        upstream_url,
        params=request.query_params,
        headers={
            k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}
        },
        content=await request.body(),
    )
    try:
        response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    headers = {
        k: v
        for k, v in response.headers.items()
        if k.lower() not in {"content-length", "transfer-encoding", "connection"}
    }
    return StreamingResponse(
        proxy_stream(response, client), status_code=response.status_code, headers=headers
    )
