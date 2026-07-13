from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class InferenceProvider(StrEnum):
    QWEN = "qwen"
    OLLAMA = "ollama"


class DevicePreference(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"


class QuantizationMode(StrEnum):
    AUTO = "auto"
    GPTQ = "gptq"
    GGUF = "gguf"
    AWQ = "awq"
    NONE = "none"


class ModelExportMode(StrEnum):
    FULL = "full"
    LORA_ONLY = "lora_only"


class ModelCatalogConfig(BaseModel):
    parameter_billions: float = Field(default=7.61, ge=7.0, le=9.0)
    base: str = "Qwen/Qwen3-8B"
    gptq: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    gguf: str = "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M"
    awq: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    tokenizer: str = "Qwen/Qwen2.5-7B-Instruct"


class OffloadConfig(BaseModel):
    enabled: bool = True
    auto_size: bool = True
    cpu_offload_gb: float = Field(default=8.0, ge=0)
    max_cpu_offload_gb: float = Field(default=64.0, ge=0)
    swap_space_gb: float = Field(default=4.0, ge=0)
    reserve_system_memory_gb: float = Field(default=4.0, ge=0)
    runtime_overhead_factor: float = Field(default=1.50, ge=1.0)
    minimum_quantization_bits: int = Field(default=4, ge=2, le=16)

    @model_validator(mode="after")
    def validate_enabled_capacity(self) -> OffloadConfig:
        if (
            self.enabled
            and self.cpu_offload_gb == 0
            and self.max_cpu_offload_gb == 0
            and self.swap_space_gb == 0
        ):
            raise ValueError("enabled offload requires cpu_offload_gb or swap_space_gb")
        return self


class VllmConfig(BaseModel):
    executable: str = "vllm"
    served_model_names: list[str] = Field(
        default_factory=lambda: [
            "local-planner",
            "local-slm-context",
            "local-slm-dispatch",
            "local-slm-runtime",
            "local-slm-kg",
        ]
    )
    dtype: str = "auto"
    max_model_len: int = Field(default=8192, ge=1)
    gpu_memory_utilization: float = Field(default=0.90, gt=0, le=1)
    tensor_parallel_size: int | None = Field(default=None, ge=1)
    trust_remote_code: bool = False
    enforce_eager: bool = False
    extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_served_names(self) -> VllmConfig:
        if not self.served_model_names:
            raise ValueError("served_model_names cannot be empty")
        return self


class OllamaConfig(BaseModel):
    executable: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:8b"
    parameter_billions: float = Field(default=8.2, ge=7.0, le=9.0)
    launch_server: bool = False
    startup_timeout_seconds: float = Field(default=30.0, gt=0)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    keep_alive: str = "10m"
    think: bool | str | None = False
    num_gpu: int | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    model: str = "qwen3-embedding:0.6b"
    dimensions: int = Field(default=1024, gt=0)
    keep_alive: str = "10m"
    truncate: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class TokenizerConfig(BaseModel):
    model: str = "Qwen/Qwen3-8B"
    add_special_tokens: bool = False


class InferenceServerConfig(BaseModel):
    host: str = "0.0.0.0"
    advertised_host: str = "127.0.0.1"
    port: int = Field(default=30000, ge=1, le=65535)
    log_level: str = "info"


class ModelServerEndpointConfig(BaseModel):
    public_url: str

    @model_validator(mode="after")
    def validate_public_url(self) -> ModelServerEndpointConfig:
        if not self.public_url.startswith(("http://", "https://")):
            raise ValueError("public_url must use http:// or https://")
        self.public_url = self.public_url.rstrip("/")
        return self


class ModelServerConfig(BaseModel):
    enabled: bool = False
    model: str = "qwen-local"
    admin_token_env: str = "MODEL_ADMIN_TOKEN"
    export_mode: ModelExportMode = ModelExportMode.FULL
    max_loras: int = Field(default=4, ge=1)
    max_cpu_loras: int = Field(default=8, ge=1)
    activation_extra_args: list[str] = Field(default_factory=list)
    cpu: ModelServerEndpointConfig = Field(
        default_factory=lambda: ModelServerEndpointConfig(public_url="http://127.0.0.1:8001")
    )
    cuda: ModelServerEndpointConfig = Field(
        default_factory=lambda: ModelServerEndpointConfig(public_url="http://127.0.0.1:8002")
    )

    @model_validator(mode="after")
    def validate_lora_capacity(self) -> ModelServerConfig:
        if self.max_cpu_loras < self.max_loras:
            raise ValueError("model_server.max_cpu_loras must be >= max_loras")
        return self


class InferenceConfig(BaseModel):
    provider: InferenceProvider = InferenceProvider.OLLAMA
    device: DevicePreference = DevicePreference.AUTO
    quantization: QuantizationMode = QuantizationMode.AUTO
    models: ModelCatalogConfig = Field(default_factory=ModelCatalogConfig)
    offload: OffloadConfig = Field(default_factory=OffloadConfig)
    vllm: VllmConfig = Field(default_factory=VllmConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    model_server: ModelServerConfig = Field(default_factory=ModelServerConfig)
    server: InferenceServerConfig = Field(default_factory=InferenceServerConfig)

    @model_validator(mode="after")
    def validate_provider_quantization(self) -> InferenceConfig:
        if self.model_server.enabled and self.provider != InferenceProvider.QWEN:
            raise ValueError("model_server hosting requires provider=qwen")
        if self.provider == InferenceProvider.OLLAMA and self.quantization in {
            QuantizationMode.GPTQ,
            QuantizationMode.AWQ,
        }:
            raise ValueError(
                "Ollama does not load GPTQ/AWQ checkpoints; choose provider=qwen "
                "or select an Ollama GGUF model tag"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> InferenceConfig:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        return cls.model_validate(payload)

    @classmethod
    def from_default_path(cls, repo_root: Path | None = None) -> InferenceConfig:
        configured = os.getenv("DULLAHAN_INFERENCE_CONFIG")
        path = (
            Path(configured) if configured else (repo_root or Path.cwd()) / "configs/inference.yaml"
        )
        return cls.from_yaml(path)
