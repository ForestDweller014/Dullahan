from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

from dullahan_inference.config import DevicePreference


@dataclass(frozen=True)
class DeviceInventory:
    device: DevicePreference | str
    cuda_device_count: int = 0
    system_memory_gb: float | None = None
    accelerator_memory_gb: float | None = None
    detection_source: str = "configured"

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", DevicePreference(self.device))


def _torch_cuda_inventory() -> DeviceInventory | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    memory_gb = sum(
        float(torch.cuda.get_device_properties(index).total_memory) / 1_000_000_000
        for index in range(max(1, int(torch.cuda.device_count())))
    )
    return DeviceInventory(
        device=DevicePreference.CUDA,
        cuda_device_count=max(1, int(torch.cuda.device_count())),
        system_memory_gb=_system_memory_gb(),
        accelerator_memory_gb=memory_gb,
        detection_source="torch",
    )


def _nvidia_smi_inventory() -> DeviceInventory | None:
    if shutil.which("nvidia-smi") is None:
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    devices = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not devices:
        return None
    memory_mib = sum(float(line.rsplit(",", 1)[1].strip()) for line in devices)
    return DeviceInventory(
        device=DevicePreference.CUDA,
        cuda_device_count=len(devices),
        system_memory_gb=_system_memory_gb(),
        accelerator_memory_gb=memory_mib * 1024**2 / 1_000_000_000,
        detection_source="nvidia-smi",
    )


def _system_memory_gb() -> float | None:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip()) / 1_000_000_000
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1_000_000_000
    except (AttributeError, OSError, ValueError):
        return None


def _apple_metal_inventory() -> DeviceInventory | None:
    if sys.platform != "darwin" or not shutil.which("system_profiler"):
        return None
    result = subprocess.run(
        ["system_profiler", "SPDisplaysDataType"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0 or "Metal Support:" not in result.stdout:
        return None
    memory_gb = _system_memory_gb()
    return DeviceInventory(
        device=DevicePreference.METAL,
        system_memory_gb=memory_gb,
        accelerator_memory_gb=memory_gb,
        detection_source="apple-metal",
    )


def detect_device(preference: DevicePreference) -> DeviceInventory:
    if preference == DevicePreference.CPU:
        return DeviceInventory(
            device=DevicePreference.CPU,
            system_memory_gb=_system_memory_gb(),
        )
    if preference == DevicePreference.METAL:
        return _apple_metal_inventory() or DeviceInventory(
            device=DevicePreference.METAL,
            system_memory_gb=_system_memory_gb(),
            detection_source="configured-unverified",
        )
    if preference == DevicePreference.CUDA:
        detected = _torch_cuda_inventory() or _nvidia_smi_inventory()
        return detected or DeviceInventory(
            device=DevicePreference.CUDA,
            cuda_device_count=1,
            system_memory_gb=_system_memory_gb(),
            detection_source="configured-unverified",
        )
    return (
        _torch_cuda_inventory()
        or _nvidia_smi_inventory()
        or _apple_metal_inventory()
        or DeviceInventory(
            device=DevicePreference.CPU,
            system_memory_gb=_system_memory_gb(),
            detection_source="fallback",
        )
    )
