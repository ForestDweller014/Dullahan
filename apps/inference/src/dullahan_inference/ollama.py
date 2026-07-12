from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dullahan_inference.config import InferenceConfig
from dullahan_inference.plan import ResolvedInferencePlan


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaResult:
    text: str
    completion_tokens: int
    prompt_tokens: int


class OllamaClient:
    def __init__(self, config: InferenceConfig, plan: ResolvedInferencePlan) -> None:
        self.config = config
        self.plan = plan

    def complete(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float | None = None,
    ) -> OllamaResult:
        options = dict(self.config.ollama.options)
        options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if self.plan.device.value == "cpu":
            options["num_gpu"] = 0
        elif self.config.ollama.num_gpu is not None:
            options["num_gpu"] = self.config.ollama.num_gpu

        payload = {
            "model": self.config.ollama.model,
            "prompt": prompt,
            "stream": False,
            "think": self.config.ollama.think,
            "keep_alive": self.config.ollama.keep_alive,
            "options": options,
        }
        request = Request(
            f"{self.config.ollama.base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.ollama.request_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise OllamaError(f"Ollama request failed: {exc.reason}") from exc
        return OllamaResult(
            text=str(data.get("response", "")).strip(),
            completion_tokens=int(data.get("eval_count", 0)),
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
        )


class OllamaProcess:
    def __init__(self, config: InferenceConfig, plan: ResolvedInferencePlan) -> None:
        self.config = config
        self.plan = plan
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if not self.config.ollama.launch_server:
            return
        environment = os.environ.copy()
        environment.update(self.plan.environment)
        self.process = subprocess.Popen(self.plan.command, env=environment)
        self._wait_until_ready()

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.config.ollama.startup_timeout_seconds
        url = f"{self.config.ollama.base_url.rstrip('/')}/api/tags"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise OllamaError("Ollama exited before becoming ready")
            try:
                with urlopen(url, timeout=1):
                    return
            except URLError:
                time.sleep(0.25)
        raise OllamaError("timed out waiting for Ollama to become ready")
