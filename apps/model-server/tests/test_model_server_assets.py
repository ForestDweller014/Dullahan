from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


# Verifies that exactly two compose variants exist.
def test_exactly_two_compose_variants_exist() -> None:
    assert {path.name for path in ROOT.glob("compose.*.yaml")} == {
        "compose.cpu.yaml",
        "compose.cuda.yaml",
    }


# Verifies that each Compose variant declares the backend matching its filename.
def test_compose_variants_declare_matching_backends() -> None:
    for backend in ("cpu", "cuda"):
        compose = yaml.safe_load((ROOT / f"compose.{backend}.yaml").read_text())
        service = next(iter(compose["services"].values()))
        assert service["environment"]["DULLAHAN_BACKEND"] == backend
        assert service["environment"]["VLLM_MAX_LORAS"] == "${VLLM_MAX_LORAS:-4}"
        assert service["environment"]["VLLM_MAX_CPU_LORAS"] == "${VLLM_MAX_CPU_LORAS:-8}"
        assert service["build"]["dockerfile"] == f"Dockerfile.{backend}"
        assert service["volumes"][0] == f"{backend}-models:/models"
        assert service["volumes"][1] == "hf-cache:/root/.cache/huggingface"


# Verifies that local env is excluded from build context.
def test_local_env_is_excluded_from_build_context() -> None:
    assert ".env" in (ROOT / ".dockerignore").read_text().splitlines()
