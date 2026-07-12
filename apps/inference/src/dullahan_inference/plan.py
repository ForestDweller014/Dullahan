from __future__ import annotations

from pydantic import BaseModel, Field

from dullahan_inference.config import (
    DevicePreference,
    InferenceConfig,
    InferenceProvider,
    QuantizationMode,
)
from dullahan_inference.device import DeviceInventory, detect_device


class ResolvedInferencePlan(BaseModel):
    provider: InferenceProvider
    engine: str
    device: DevicePreference
    device_detection_source: str
    cuda_device_count: int = 0
    quantization: QuantizationMode
    model: str
    tokenizer: str | None = None
    api_base_url: str
    command: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    offload_enabled: bool
    quantization_bits: int
    estimated_model_memory_gb: float
    system_memory_gb: float | None = None
    accelerator_memory_gb: float | None = None
    usable_system_memory_gb: float | None = None
    cpu_offload_gb: float = 0
    memory_fit: bool = True
    container_backend: str | None = None
    admin_base_url: str | None = None
    admin_token_env: str | None = None
    model_export_mode: str = "full"
    max_loras: int = 4
    max_cpu_loras: int = 8
    activation_extra_args: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _quantization_bits(mode: QuantizationMode) -> int:
    return 16 if mode == QuantizationMode.NONE else 4


def _estimated_model_memory_gb(
    config: InferenceConfig,
    quantization: QuantizationMode,
) -> float:
    parameter_billions = (
        config.ollama.parameter_billions
        if config.provider == InferenceProvider.OLLAMA
        else config.models.parameter_billions
    )
    weight_gb = parameter_billions * _quantization_bits(quantization) / 8
    return round(weight_gb * config.offload.runtime_overhead_factor, 2)


def _usable_system_memory_gb(
    config: InferenceConfig,
    inventory: DeviceInventory,
) -> float | None:
    if inventory.system_memory_gb is None:
        return None
    return round(
        max(0.0, inventory.system_memory_gb - config.offload.reserve_system_memory_gb),
        2,
    )


def _memory_plan(
    config: InferenceConfig,
    inventory: DeviceInventory,
    quantization: QuantizationMode,
) -> tuple[float, bool, list[str]]:
    estimated_gb = _estimated_model_memory_gb(config, quantization)
    usable_system_gb = _usable_system_memory_gb(config, inventory)
    notes: list[str] = []

    if _quantization_bits(quantization) < config.offload.minimum_quantization_bits:
        return (
            0.0,
            False,
            ["The selected checkpoint is more aggressively quantized than the configured minimum."],
        )

    if inventory.device != DevicePreference.CUDA:
        fits = usable_system_gb is None or estimated_gb <= usable_system_gb
        if not fits:
            notes.append(
                f"Estimated model demand is {estimated_gb} GB, but only "
                f"{usable_system_gb} GB of system memory is usable."
            )
        return 0.0, fits, notes

    if inventory.accelerator_memory_gb is None:
        fallback = config.offload.cpu_offload_gb if config.offload.enabled else 0.0
        notes.append(
            "CUDA memory could not be measured; using the configured CPU offload capacity."
        )
        return fallback, True, notes

    usable_accelerator_gb = inventory.accelerator_memory_gb * config.vllm.gpu_memory_utilization
    total_shortfall_gb = max(0.0, estimated_gb - usable_accelerator_gb)
    if total_shortfall_gb == 0:
        return 0.0, True, notes
    if not config.offload.enabled:
        notes.append(
            f"Estimated model demand exceeds usable VRAM by {round(total_shortfall_gb, 2)} GB, "
            "and CPU offload is disabled."
        )
        return 0.0, False, notes

    device_count = max(1, inventory.cuda_device_count)
    per_device_shortfall_gb = total_shortfall_gb / device_count
    requested_gb = (
        per_device_shortfall_gb if config.offload.auto_size else config.offload.cpu_offload_gb
    )
    cpu_offload_gb = min(requested_gb, config.offload.max_cpu_offload_gb)
    total_host_demand_gb = cpu_offload_gb * device_count
    fits_host = usable_system_gb is None or total_host_demand_gb <= usable_system_gb
    fits = cpu_offload_gb >= per_device_shortfall_gb and fits_host
    if config.offload.auto_size:
        notes.append(
            f"Quantization stops at {_quantization_bits(quantization)}-bit; "
            f"{round(cpu_offload_gb, 2)} GB per CUDA device is assigned to CPU offload "
            "for the VRAM shortfall."
        )
    if not fits:
        notes.append(
            "The configured host-offload capacity cannot absorb the remaining VRAM shortfall."
        )
    return round(cpu_offload_gb, 2), fits, notes


def _resolve_quantization(
    config: InferenceConfig,
    inventory: DeviceInventory,
) -> QuantizationMode:
    if config.quantization != QuantizationMode.AUTO:
        return config.quantization
    if config.provider == InferenceProvider.OLLAMA:
        return QuantizationMode.GGUF
    return (
        QuantizationMode.GPTQ
        if inventory.device == DevicePreference.CUDA
        else QuantizationMode.GGUF
    )


def _model_for_quantization(config: InferenceConfig, mode: QuantizationMode) -> str:
    return {
        QuantizationMode.GPTQ: config.models.gptq,
        QuantizationMode.GGUF: config.models.gguf,
        QuantizationMode.AWQ: config.models.awq,
        QuantizationMode.NONE: config.models.base,
    }[mode]


def _vllm_plan(
    config: InferenceConfig,
    inventory: DeviceInventory,
    quantization: QuantizationMode,
) -> ResolvedInferencePlan:
    model = _model_for_quantization(config, quantization)
    cpu_offload_gb, memory_fit, memory_notes = _memory_plan(config, inventory, quantization)
    vllm_device = (
        DevicePreference.CPU if inventory.device == DevicePreference.METAL else inventory.device
    )
    command = [
        config.vllm.executable,
        "serve",
        model,
        "--host",
        config.server.host,
        "--port",
        str(config.server.port),
        "--device",
        vllm_device.value,
        "--dtype",
        config.vllm.dtype,
        "--max-model-len",
        str(config.vllm.max_model_len),
        "--served-model-name",
        *config.vllm.served_model_names,
    ]
    if quantization != QuantizationMode.NONE:
        command.extend(["--quantization", quantization.value])
    tokenizer = config.models.tokenizer if quantization == QuantizationMode.GGUF else None
    if tokenizer:
        command.extend(["--tokenizer", tokenizer])
    if config.offload.enabled and config.offload.swap_space_gb > 0:
        command.extend(["--swap-space", str(config.offload.swap_space_gb)])
    if inventory.device == DevicePreference.CUDA:
        command.extend(["--gpu-memory-utilization", str(config.vllm.gpu_memory_utilization)])
        if cpu_offload_gb > 0:
            command.extend(["--cpu-offload-gb", str(cpu_offload_gb)])
        tensor_parallel_size = config.vllm.tensor_parallel_size or inventory.cuda_device_count
        if tensor_parallel_size > 1:
            command.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if config.vllm.trust_remote_code:
        command.append("--trust-remote-code")
    if config.vllm.enforce_eager:
        command.append("--enforce-eager")
    command.extend(config.vllm.extra_args)

    notes = list(memory_notes)
    if inventory.device == DevicePreference.CPU:
        notes.append("CPU execution keeps the complete model in system memory.")
    if inventory.device == DevicePreference.METAL:
        notes.append(
            "vLLM has no native Metal device; use provider=ollama for Apple GPU execution."
        )
    if quantization == QuantizationMode.GGUF:
        notes.append("GGUF serving requires the vllm-gguf-plugin and remains experimental.")
    return ResolvedInferencePlan(
        provider=config.provider,
        engine="vllm",
        device=inventory.device,
        device_detection_source=inventory.detection_source,
        cuda_device_count=inventory.cuda_device_count,
        quantization=quantization,
        model=model,
        tokenizer=tokenizer,
        api_base_url=f"http://{config.server.advertised_host}:{config.server.port}/v1",
        command=command,
        offload_enabled=config.offload.enabled,
        quantization_bits=_quantization_bits(quantization),
        estimated_model_memory_gb=_estimated_model_memory_gb(config, quantization),
        system_memory_gb=inventory.system_memory_gb,
        accelerator_memory_gb=inventory.accelerator_memory_gb,
        usable_system_memory_gb=_usable_system_memory_gb(config, inventory),
        cpu_offload_gb=cpu_offload_gb,
        memory_fit=memory_fit,
        notes=notes,
    )


def _ollama_plan(
    config: InferenceConfig,
    inventory: DeviceInventory,
    quantization: QuantizationMode,
) -> ResolvedInferencePlan:
    _, memory_fit, memory_notes = _memory_plan(config, inventory, quantization)
    environment = {}
    if inventory.device == DevicePreference.CPU:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    command = [config.ollama.executable, "serve"] if config.ollama.launch_server else []
    return ResolvedInferencePlan(
        provider=config.provider,
        engine="ollama",
        device=inventory.device,
        device_detection_source=inventory.detection_source,
        cuda_device_count=inventory.cuda_device_count,
        quantization=quantization,
        model=config.ollama.model,
        api_base_url=f"http://{config.server.advertised_host}:{config.server.port}/v1",
        command=command,
        environment=environment,
        offload_enabled=config.offload.enabled,
        quantization_bits=_quantization_bits(quantization),
        estimated_model_memory_gb=_estimated_model_memory_gb(config, quantization),
        system_memory_gb=inventory.system_memory_gb,
        accelerator_memory_gb=inventory.accelerator_memory_gb,
        usable_system_memory_gb=_usable_system_memory_gb(config, inventory),
        memory_fit=memory_fit,
        notes=memory_notes
        + ["Ollama controls layer placement; select a quantized model tag in ollama.model."],
    )


def _model_server_plan(
    config: InferenceConfig,
    inventory: DeviceInventory,
    quantization: QuantizationMode,
) -> ResolvedInferencePlan:
    backend = "cuda" if inventory.device == DevicePreference.CUDA else "cpu"
    target_device = DevicePreference.CUDA if backend == "cuda" else DevicePreference.CPU
    endpoint = getattr(config.model_server, backend)
    return ResolvedInferencePlan(
        provider=config.provider,
        engine="vllm-model-server",
        device=target_device,
        device_detection_source=inventory.detection_source,
        cuda_device_count=inventory.cuda_device_count if backend == "cuda" else 0,
        quantization=quantization,
        model=config.model_server.model,
        api_base_url=f"{endpoint.public_url}/v1",
        offload_enabled=config.offload.enabled,
        quantization_bits=_quantization_bits(quantization),
        estimated_model_memory_gb=_estimated_model_memory_gb(config, quantization),
        memory_fit=True,
        container_backend=backend,
        admin_base_url=endpoint.public_url,
        admin_token_env=config.model_server.admin_token_env,
        model_export_mode=config.model_server.export_mode.value,
        max_loras=config.model_server.max_loras,
        max_cpu_loras=config.model_server.max_cpu_loras,
        activation_extra_args=config.model_server.activation_extra_args,
        notes=[
            "The external model-server container owns hardware capacity checks and vLLM lifecycle.",
            "Activate the configured model through the admin API before sending /v1 requests.",
        ],
    )


def resolve_inference_plan(
    config: InferenceConfig,
    *,
    inventory: DeviceInventory | None = None,
) -> ResolvedInferencePlan:
    inventory = inventory or detect_device(config.device)
    quantization = _resolve_quantization(config, inventory)
    if config.model_server.enabled:
        return _model_server_plan(config, inventory, quantization)
    if config.provider == InferenceProvider.QWEN:
        return _vllm_plan(config, inventory, quantization)
    return _ollama_plan(config, inventory, quantization)
