import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SERVER = ROOT / "apps" / "model-server"


def test_python_distribution_declares_apache_license() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["license"] == "Apache-2.0"
    assert metadata["project"]["license-files"] == [
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
    ]
    assert "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert (ROOT / "NOTICE").is_file()
    assert (ROOT / "THIRD_PARTY_NOTICES.md").is_file()


def test_model_server_images_include_license_materials() -> None:
    assert (MODEL_SERVER / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    for name in ("NOTICE", "THIRD_PARTY_NOTICES.md"):
        assert (MODEL_SERVER / name).is_file()

    for dockerfile in ("Dockerfile.cpu", "Dockerfile.cuda"):
        contents = (MODEL_SERVER / dockerfile).read_text(encoding="utf-8")
        assert "COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md /licenses/dullahan/" in contents


def test_model_artifacts_are_documented_as_separately_licensed() -> None:
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "does not include model weights" in notices
    assert "not relicensed by Dullahan" in notices
    assert "Qwen/Qwen3-8B" in notices
