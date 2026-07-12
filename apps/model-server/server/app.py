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
    FULL = "full"
    LORA_ONLY = "lora_only"


class ExportMode(StrEnum):
    FULL = "full"
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
    package_mode: PackageMode = PackageMode.FULL
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
DEFAULT_EXPORT_MODE = ExportMode(os.getenv("MODEL_EXPORT_MODE", ExportMode.FULL))
ADAPTERS_DIRECTORY = "adapters"
ADAPTER_WEIGHT_PATTERNS = ("*.safetensors", "*.bin")
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
        config_path = adapter / "adapter_config.json"
        if not config_path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"LoRA adapter {adapter.name!r} is missing adapter_config.json",
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
                detail=f"LoRA adapter {adapter.name!r} has no supported weight file",
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result.append(
            {
                "name": checked_name(adapter.name),
                "directory": f"{ADAPTERS_DIRECTORY}/{adapter.name}",
                "base_model": config.get("base_model_name_or_path"),
                "bytes": dir_size(adapter),
            }
        )
    return result


def package_mode(path: Path, manifest: dict[str, Any] | None = None) -> PackageMode:
    manifest = manifest if manifest is not None else model_manifest(path)
    try:
        return PackageMode(manifest.get("package_mode", PackageMode.FULL))
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
        "adapters": adapters,
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
    if package_mode(path, manifest) == PackageMode.LORA_ONLY:
        base_model = manifest.get("base_model")
        if not base_model:
            raise HTTPException(status_code=400, detail="LoRA-only package has no base_model")
        return str(base_model)
    if manifest.get("quantization") != ModelQuantization.GGUF:
        return path
    candidates = sorted(path.glob("*.gguf"))
    if len(candidates) != 1:
        raise HTTPException(
            status_code=400,
            detail="GGUF model directories must contain exactly one top-level .gguf file",
        )
    return candidates[0]


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


def validate_model_directory(path: Path) -> None:
    if not (path / "config.json").is_file():
        raise HTTPException(
            status_code=400,
            detail="Archive is not a Hugging Face model directory: config.json is missing",
        )
    weight_candidates = (
        list(path.glob("*.safetensors"))
        + list(path.glob("*.bin"))
        + list(path.glob("*.gguf"))
        + list(path.glob("*.safetensors.index.json"))
        + list(path.glob("*.bin.index.json"))
    )
    if not weight_candidates:
        raise HTTPException(status_code=400, detail="No supported model weight files were found")


def validate_model_package(path: Path) -> dict[str, Any]:
    manifest = model_manifest(path)
    mode = package_mode(path, manifest)
    adapters = lora_adapters(path)
    if mode == PackageMode.FULL:
        validate_model_directory(path)
    else:
        if not manifest.get("base_model"):
            raise HTTPException(
                status_code=400,
                detail="LoRA-only packages must declare base_model in dullahan-model.json",
            )
        if not adapters:
            raise HTTPException(
                status_code=400,
                detail="LoRA-only packages must contain at least one LoRA adapter",
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
        if spec.package_mode == PackageMode.LORA_ONLY:
            if not spec.base_model:
                raise HTTPException(
                    status_code=400,
                    detail="LoRA-only Hugging Face imports require base_model",
                )
            package = Path(tempfile.mkdtemp(prefix=f".{name}-package-", dir=MODEL_ROOT))
            adapter = package / ADAPTERS_DIRECTORY / checked_name(spec.adapter_name or name)
            adapter.parent.mkdir()
            temp.rename(adapter)
            temp = package
        else:
            validate_model_directory(temp)
        prepare_manifest(
            temp,
            name=name,
            source="huggingface",
            repo_id=spec.repo_id,
            revision=spec.revision,
            base_model=spec.base_model or spec.repo_id,
            quantization=spec.quantization,
            supported_backends=spec.supported_backends,
            package_mode=spec.package_mode.value,
        )
        validate_model_package(temp)
        backup = target.with_name(f".{name}.old")
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            target.rename(backup)
        temp.rename(target)
        if backup.exists():
            shutil.rmtree(backup)
        return {"name": name, "bytes": dir_size(target)}
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
        )

        staging = MODEL_ROOT / f".{name}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        source.rename(staging)
        backup = MODEL_ROOT / f".{name}.old"
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            target.rename(backup)
        staging.rename(target)
        if backup.exists():
            shutil.rmtree(backup)
        return {"name": name, "sha256": digest.hexdigest(), "bytes": dir_size(target)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


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
    try:
        mode = ExportMode(requested_mode) if requested_mode else DEFAULT_EXPORT_MODE
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="mode must be full or lora_only") from exc

    export_dir = MODEL_ROOT / ".exports"
    export_dir.mkdir(exist_ok=True)
    suffix = "" if mode == ExportMode.FULL else "-lora-only"
    output = export_dir / f"{name}{suffix}.tar.gz"
    temp_output = export_dir / f".{name}{suffix}.tar.gz.tmp"
    if temp_output.exists():
        temp_output.unlink()
    export_source = source
    staging: Path | None = None
    if mode == ExportMode.LORA_ONLY:
        manifest = model_manifest(source)
        adapters = lora_adapters(source)
        if not adapters:
            raise HTTPException(status_code=409, detail="Model has no LoRA adapters to export")
        if not manifest.get("base_model"):
            raise HTTPException(
                status_code=409,
                detail="Model metadata has no named base_model for a LoRA-only export",
            )
        staging = Path(tempfile.mkdtemp(prefix=f".{name}-lora-export-", dir=export_dir))
        export_source = staging / name
        export_source.mkdir()
        shutil.copytree(source / ADAPTERS_DIRECTORY, export_source / ADAPTERS_DIRECTORY)
        write_manifest(
            export_source,
            {
                **manifest,
                "name": name,
                "package_mode": PackageMode.LORA_ONLY.value,
                "adapters": lora_adapters(export_source),
                "exported_from": manifest.get("package_mode", PackageMode.FULL.value),
            },
        )
    try:
        with tarfile.open(temp_output, "w:gz") as tf:
            tf.add(export_source, arcname=name, recursive=True)
    finally:
        if staging:
            shutil.rmtree(staging, ignore_errors=True)
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
