from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
import yaml
from dullahan_inference.config import InferenceConfig
from dullahan_inference.device import detect_device
from dullahan_inference.plan import resolve_inference_plan


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.mark.local_inference
def test_local_ollama_generates_a_basic_response(tmp_path) -> None:
    if os.getenv("DULLAHAN_RUN_LOCAL_INFERENCE") != "1":
        pytest.skip("set DULLAHAN_RUN_LOCAL_INFERENCE=1 to run a local model")

    port = _free_port()
    config = InferenceConfig(
        provider="ollama",
        ollama={"model": os.getenv("DULLAHAN_TEST_MODEL", "qwen3:8b")},
        server={"host": "127.0.0.1", "advertised_host": "127.0.0.1", "port": port},
    )
    plan = resolve_inference_plan(config, inventory=detect_device(config.device))
    assert plan.memory_fit, plan.notes
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(source_root), environment.get("PYTHONPATH"))
        if value
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "dullahan_inference.cli",
            "serve",
            "--config",
            str(config_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"inference server exited early: {process.stderr.read()}")
        try:
            with urlopen(health_url, timeout=1):
                break
        except URLError:
            time.sleep(0.1)
    else:
        process.terminate()
        pytest.fail("inference server did not become healthy")

    request = Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(
            {
                "model": "local-planner",
                "prompt": "Reply with exactly: inference-ok",
                "max_tokens": 32,
                "temperature": 0,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert payload["choices"][0]["text"]
    assert payload["usage"]["completion_tokens"] > 0
