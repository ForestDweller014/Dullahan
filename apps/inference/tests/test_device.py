from __future__ import annotations

from dullahan_inference.config import DevicePreference
from dullahan_inference.device import DeviceInventory, detect_device


def test_auto_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr("dullahan_inference.device._torch_cuda_inventory", lambda: None)
    monkeypatch.setattr("dullahan_inference.device._nvidia_smi_inventory", lambda: None)
    monkeypatch.setattr("dullahan_inference.device._apple_metal_inventory", lambda: None)

    inventory = detect_device(DevicePreference.AUTO)

    assert inventory.device == DevicePreference.CPU
    assert inventory.detection_source == "fallback"


def test_explicit_cuda_is_preserved_when_detection_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("dullahan_inference.device._torch_cuda_inventory", lambda: None)
    monkeypatch.setattr("dullahan_inference.device._nvidia_smi_inventory", lambda: None)

    inventory = detect_device(DevicePreference.CUDA)

    assert inventory.device == DevicePreference.CUDA
    assert inventory.cuda_device_count == 1
    assert inventory.detection_source == "configured-unverified"


def test_device_inventory_normalizes_string_values() -> None:
    assert DeviceInventory(device="cuda").device == DevicePreference.CUDA


def test_auto_device_uses_apple_metal_when_available(monkeypatch) -> None:
    metal = DeviceInventory(
        device="metal",
        system_memory_gb=18.0,
        accelerator_memory_gb=18.0,
        detection_source="apple-metal",
    )
    monkeypatch.setattr("dullahan_inference.device._torch_cuda_inventory", lambda: None)
    monkeypatch.setattr("dullahan_inference.device._nvidia_smi_inventory", lambda: None)
    monkeypatch.setattr("dullahan_inference.device._apple_metal_inventory", lambda: metal)

    assert detect_device(DevicePreference.AUTO) == metal
