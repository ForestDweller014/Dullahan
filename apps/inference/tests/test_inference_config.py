from __future__ import annotations

from pathlib import Path

import pytest
from dullahan_inference.config import (
    DevicePreference,
    InferenceConfig,
    QuantizationMode,
)
from dullahan_inference.device import DeviceInventory
from dullahan_inference.plan import resolve_inference_plan
from pydantic import ValidationError


def inventory(device: DevicePreference, count: int = 0) -> DeviceInventory:
    return DeviceInventory(
        device=device,
        cuda_device_count=count,
        detection_source="test",
    )


# Verifies that default config selects GGUF on CPU.
def test_default_config_selects_gguf_on_cpu() -> None:
    config = InferenceConfig(provider="qwen")

    plan = resolve_inference_plan(config, inventory=inventory(DevicePreference.CPU))

    assert plan.engine == "vllm"
    assert plan.quantization == QuantizationMode.GGUF
    assert plan.model == config.models.gguf
    assert plan.tokenizer == config.models.tokenizer
    assert "--quantization" in plan.command
    assert "gguf" in plan.command
    assert "--cpu-offload-gb" not in plan.command
    assert "--swap-space" in plan.command


# Verifies that default config selects GPTQ and offload on CUDA.
def test_default_config_selects_gptq_and_offload_on_cuda() -> None:
    config = InferenceConfig(provider="qwen")

    plan = resolve_inference_plan(config, inventory=inventory(DevicePreference.CUDA, 2))

    assert plan.quantization == QuantizationMode.GPTQ
    assert plan.model == config.models.gptq
    assert plan.cuda_device_count == 2
    assert plan.command[plan.command.index("--cpu-offload-gb") + 1] == "8.0"
    assert plan.command[plan.command.index("--tensor-parallel-size") + 1] == "2"
    served_name_index = plan.command.index("--served-model-name")
    assert plan.command[served_name_index + 1] == "local-planner"


# Verifies that explicit AWQ override wins on CPU.
def test_explicit_awq_override_wins_on_cpu() -> None:
    config = InferenceConfig(provider="qwen", quantization=QuantizationMode.AWQ)

    plan = resolve_inference_plan(config, inventory=inventory(DevicePreference.CPU))

    assert plan.quantization == QuantizationMode.AWQ
    assert plan.model == config.models.awq


# Verifies that disabling offload removes CPU-offload and swap-space launch flags.
def test_disabling_offload_removes_host_memory_flags() -> None:
    config = InferenceConfig(provider="qwen", offload={"enabled": False})

    plan = resolve_inference_plan(config, inventory=inventory(DevicePreference.CUDA, 1))

    assert "--cpu-offload-gb" not in plan.command
    assert "--swap-space" not in plan.command
    assert plan.offload_enabled is False


# Verifies that Ollama rejects GPTQ and AWQ.
def test_ollama_rejects_gptq_and_awq() -> None:
    with pytest.raises(ValidationError, match="Ollama does not load GPTQ/AWQ"):
        InferenceConfig(provider="ollama", quantization="gptq")


# Verifies that repository config is valid.
def test_repository_config_is_valid() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    config = InferenceConfig.from_yaml(repo_root / "configs/inference.yaml")

    assert 7 <= config.models.parameter_billions <= 9
    assert config.offload.enabled is True
    assert config.offload.minimum_quantization_bits == 4
    assert config.provider.value == "ollama"
    assert config.embeddings.model == "qwen3-embedding:0.6b"
    assert config.embeddings.dimensions == 1024
    assert config.tokenizer.model == "Qwen/Qwen3-8B"


# Verifies that CUDA shortfall uses offload instead of more quantization.
def test_cuda_shortfall_uses_offload_instead_of_more_quantization() -> None:
    config = InferenceConfig(provider="qwen")
    hardware = DeviceInventory(
        device="cuda",
        cuda_device_count=1,
        system_memory_gb=16,
        accelerator_memory_gb=4,
        detection_source="test",
    )

    plan = resolve_inference_plan(config, inventory=hardware)

    assert plan.quantization == QuantizationMode.GPTQ
    assert plan.quantization_bits == 4
    assert plan.cpu_offload_gb == pytest.approx(2.11, abs=0.01)
    assert plan.memory_fit is True
    assert plan.command[plan.command.index("--cpu-offload-gb") + 1] == "2.11"


# Verifies that CUDA plan fails when host cannot absorb VRAM shortfall.
def test_cuda_plan_fails_when_host_cannot_absorb_vram_shortfall() -> None:
    config = InferenceConfig(
        provider="qwen",
        offload={"reserve_system_memory_gb": 3.5},
    )
    hardware = DeviceInventory(
        device="cuda",
        cuda_device_count=1,
        system_memory_gb=4,
        accelerator_memory_gb=2,
        detection_source="test",
    )

    plan = resolve_inference_plan(config, inventory=hardware)

    assert plan.cpu_offload_gb > plan.usable_system_memory_gb
    assert plan.memory_fit is False


# Verifies that CPU plan fails early when quantized model exceeds RAM.
def test_cpu_plan_fails_early_when_quantized_model_exceeds_ram() -> None:
    config = InferenceConfig(
        provider="qwen",
        offload={"reserve_system_memory_gb": 2},
    )
    hardware = DeviceInventory(
        device="cpu",
        system_memory_gb=5,
        detection_source="test",
    )

    plan = resolve_inference_plan(config, inventory=hardware)

    assert plan.quantization_bits == 4
    assert plan.estimated_model_memory_gb == 5.71
    assert plan.memory_fit is False


# Verifies that metal Ollama plan uses unified memory.
def test_metal_ollama_plan_uses_unified_memory() -> None:
    config = InferenceConfig(provider="ollama")
    hardware = DeviceInventory(
        device="metal",
        system_memory_gb=18,
        accelerator_memory_gb=18,
        detection_source="apple-metal",
    )

    plan = resolve_inference_plan(config, inventory=hardware)

    assert plan.engine == "ollama"
    assert plan.device == DevicePreference.METAL
    assert plan.memory_fit is True
    assert plan.estimated_model_memory_gb == 6.15
    assert plan.environment == {}


# Verifies that metal Qwen plan falls back to vllm CPU device.
def test_metal_qwen_plan_falls_back_to_vllm_cpu_device() -> None:
    config = InferenceConfig(provider="qwen")
    hardware = DeviceInventory(
        device="metal",
        system_memory_gb=18,
        accelerator_memory_gb=18,
        detection_source="apple-metal",
    )

    plan = resolve_inference_plan(config, inventory=hardware)

    assert plan.command[plan.command.index("--device") + 1] == "cpu"
    assert any("provider=ollama" in note for note in plan.notes)
