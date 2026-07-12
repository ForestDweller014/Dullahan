from __future__ import annotations

import json
from pathlib import Path

import pytest
from dullahan_inference.cli import run_from_args
from dullahan_inference.device import DeviceInventory


def test_plan_command_prints_resolved_configuration(capsys) -> None:
    repo_root = Path(__file__).resolve().parents[3]

    result = run_from_args(["plan", "--config", str(repo_root / "configs/inference.yaml")])

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["engine"] == "vllm"
    assert payload["quantization"] in {"gguf", "gptq"}
    assert payload["offload_enabled"] is True
    assert payload["quantization_bits"] == 4


def test_serve_rejects_a_plan_that_cannot_fit_memory(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        "provider: ollama\noffload:\n  reserve_system_memory_gb: 4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "dullahan_inference.plan.detect_device",
        lambda _: DeviceInventory(device="cpu", system_memory_gb=5),
    )

    with pytest.raises(SystemExit, match="does not fit available memory"):
        run_from_args(["serve", "--config", str(config_path)])
