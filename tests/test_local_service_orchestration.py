from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


# Verifies that docker compose defines CAL EDL and agent services.
def test_docker_compose_defines_cal_edl_and_agent_services() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert {"cal", "edl", "agent"} <= set(services)
    assert services["cal"]["command"] == ["dullahan-cal"]
    assert services["edl"]["command"] == ["dullahan-edl"]
    assert "8100:8100" in services["cal"]["ports"]
    assert "8200:8200" in services["edl"]["ports"]
    assert "--transport" in services["agent"]["command"]
    assert "http://cal:8100" in services["agent"]["command"]
    assert "http://edl:8200" in services["agent"]["command"]
    assert (
        services["agent"]["environment"]["DULLAHAN_INFERENCE_PROVIDER"]
        == "${DULLAHAN_INFERENCE_PROVIDER:-http}"
    )
    assert (
        services["agent"]["environment"]["DULLAHAN_LOCAL_INFERENCE_BASE_URL"]
        == "http://host.docker.internal:30000/v1"
    )


# Verifies that dockerfile installs project editable.
def test_dockerfile_installs_project_editable() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "pip install --no-cache-dir -e ." in dockerfile
    assert "PYTHONPATH=" in dockerfile
