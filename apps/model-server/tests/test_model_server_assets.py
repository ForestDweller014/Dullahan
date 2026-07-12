from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_exactly_two_compose_variants_exist() -> None:
    assert {path.name for path in ROOT.glob("compose.*.yaml")} == {
        "compose.cpu.yaml",
        "compose.cuda.yaml",
    }


def test_compose_variants_declare_matching_backends() -> None:
    for backend in ("cpu", "cuda"):
        compose = yaml.safe_load((ROOT / f"compose.{backend}.yaml").read_text())
        service = next(iter(compose["services"].values()))
        assert service["environment"]["DULLAHAN_BACKEND"] == backend
        assert service["environment"]["MODEL_EXPORT_MODE"] == "${MODEL_EXPORT_MODE:-full}"
        assert service["environment"]["VLLM_MAX_LORAS"] == "${VLLM_MAX_LORAS:-4}"
        assert service["environment"]["VLLM_MAX_CPU_LORAS"] == "${VLLM_MAX_CPU_LORAS:-8}"
        assert service["build"]["dockerfile"] == f"Dockerfile.{backend}"


def test_local_env_is_excluded_from_build_context() -> None:
    assert ".env" in (ROOT / ".dockerignore").read_text().splitlines()
