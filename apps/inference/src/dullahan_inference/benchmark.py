from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

NANOSECONDS = 1_000_000_000


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    prompt: str
    max_tokens: int


DEFAULT_CASES = (
    BenchmarkCase(
        name="short_factual",
        prompt="In two sentences, explain why quantization reduces model memory use.",
        max_tokens=96,
    ),
    BenchmarkCase(
        name="structured_reasoning",
        prompt=(
            "Return JSON with keys answer and steps. A server has 18 GB RAM, reserves "
            "4 GB, and a model needs 6.2 GB. State whether it fits and the remaining GB."
        ),
        max_tokens=128,
    ),
    BenchmarkCase(
        name="long_context_summary",
        prompt=(
            "Summarize the following deployment policy in four bullets, preserving all "
            "numbers: The CPU service listens on 8001 and uses GGUF. The CUDA service "
            "listens on 8002 and defaults to GPTQ. Both reserve 4 GB of host RAM, retain "
            "models in named volumes, expose an OpenAI-compatible API, and require an "
            "admin token for activation. CUDA can offload up to 64 GB per device. "
        )
        * 4,
        max_tokens=160,
    ),
    BenchmarkCase(
        name="code_generation",
        prompt=(
            "Write a compact Python function that computes tokens per second from an "
            "integer token count and nanosecond duration. Handle zero safely."
        ),
        max_tokens=160,
    ),
)


@dataclass(frozen=True)
class MemorySnapshot:
    ollama_rss_bytes: int
    system_used_bytes: int


@dataclass(frozen=True)
class BenchmarkRun:
    case: str
    iteration: int
    cold_start: bool
    wall_time_seconds: float
    time_to_first_token_seconds: float | None
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_per_second: float
    generation_tokens_per_second: float
    ollama_total_seconds: float
    model_load_seconds: float
    prompt_eval_seconds: float
    generation_seconds: float
    ollama_rss_before_bytes: int
    ollama_rss_peak_bytes: int
    ollama_rss_after_bytes: int
    system_used_before_bytes: int
    system_used_peak_bytes: int
    system_used_after_bytes: int
    model_resident_bytes: int
    accelerator_resident_bytes: int
    reported_host_offload_bytes: int
    response_characters: int
    response_sha256: str


def tokens_per_second(tokens: int, duration_ns: int) -> float:
    if tokens <= 0 or duration_ns <= 0:
        return 0.0
    return round(tokens * NANOSECONDS / duration_ns, 3)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percent * len(ordered)) - 1)
    return ordered[index]


def summarize_runs(runs: list[BenchmarkRun]) -> dict[str, float | int]:
    warm = [run for run in runs if not run.cold_start]
    selected = warm or runs
    latencies = [run.wall_time_seconds for run in selected]
    throughputs = [run.generation_tokens_per_second for run in selected]
    ttft = [
        run.time_to_first_token_seconds
        for run in selected
        if run.time_to_first_token_seconds is not None
    ]
    cases = {run.case for run in selected}
    deterministic_cases = sum(
        len({run.response_sha256 for run in selected if run.case == case}) == 1 for case in cases
    )
    return {
        "measured_runs": len(selected),
        "median_wall_time_seconds": round(median(latencies), 4) if latencies else 0.0,
        "p95_wall_time_seconds": round(percentile(latencies, 0.95), 4),
        "median_time_to_first_token_seconds": round(median(ttft), 4) if ttft else 0.0,
        "p95_time_to_first_token_seconds": round(percentile(ttft, 0.95), 4),
        "median_generation_tokens_per_second": (
            round(median(throughputs), 3) if throughputs else 0.0
        ),
        "minimum_generation_tokens_per_second": (
            round(min(throughputs), 3) if throughputs else 0.0
        ),
        "peak_ollama_rss_bytes": max((run.ollama_rss_peak_bytes for run in selected), default=0),
        "peak_system_used_bytes": max((run.system_used_peak_bytes for run in selected), default=0),
        "peak_ollama_rss_increase_bytes": max(
            (run.ollama_rss_peak_bytes - run.ollama_rss_before_bytes for run in selected),
            default=0,
        ),
        "peak_system_used_increase_bytes": max(
            (run.system_used_peak_bytes - run.system_used_before_bytes for run in selected),
            default=0,
        ),
        "deterministic_cases": deterministic_cases,
        "total_cases": len(cases),
    }


def _process_rss_bytes() -> int:
    result = subprocess.run(
        ["ps", "-axo", "rss=,command="],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    total_kib = 0
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\d+)\s+(.*)", line)
        if not match:
            continue
        command = match.group(2).lower()
        if "ollama serve" in command or "ollama runner" in command:
            total_kib += int(match.group(1))
    return total_kib * 1024


def _mac_system_used_bytes() -> int:
    result = subprocess.run(
        ["vm_stat"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    page_match = re.search(r"page size of (\d+) bytes", result.stdout)
    page_size = int(page_match.group(1)) if page_match else 4096
    pages: dict[str, int] = {}
    for line in result.stdout.splitlines()[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        pages[key] = int(value.strip().rstrip("."))
    used_pages = (
        pages.get("Pages active", 0)
        + pages.get("Pages wired down", 0)
        + pages.get("Pages occupied by compressor", 0)
    )
    return used_pages * page_size


def memory_snapshot() -> MemorySnapshot:
    system_used = _mac_system_used_bytes() if platform.system() == "Darwin" else 0
    return MemorySnapshot(
        ollama_rss_bytes=_process_rss_bytes(),
        system_used_bytes=system_used,
    )


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.10) -> None:
        self.interval_seconds = interval_seconds
        self.before = memory_snapshot()
        self.peak = self.before
        self.after = self.before
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.after = memory_snapshot()
        self._record(self.after)

    def _record(self, sample: MemorySnapshot) -> None:
        self.peak = MemorySnapshot(
            ollama_rss_bytes=max(self.peak.ollama_rss_bytes, sample.ollama_rss_bytes),
            system_used_bytes=max(self.peak.system_used_bytes, sample.system_used_bytes),
        )

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._record(memory_snapshot())


def _json_request(
    url: str,
    payload: dict | None = None,
    *,
    timeout: float = 120,
) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc.reason}") from exc


def _model_entry(data: dict, model: str) -> dict:
    for entry in data.get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            return entry
    return {}


class OllamaGGUFBenchmark:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        num_gpu: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_gpu = num_gpu

    def model_metadata(self) -> dict:
        shown = _json_request(f"{self.base_url}/api/show", {"model": self.model})
        tags = _json_request(f"{self.base_url}/api/tags")
        tag = _model_entry(tags, self.model)
        details = shown.get("details", {})
        info = shown.get("model_info", {})
        parameter_count = int(info.get("general.parameter_count", 0))
        storage_bytes = int(tag.get("size", 0))
        return {
            "name": self.model,
            "format": details.get("format"),
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "parameter_count": parameter_count,
            "quantization_level": details.get("quantization_level"),
            "quantization_version": info.get("general.quantization_version"),
            "storage_bytes": storage_bytes,
            "effective_storage_bits_per_parameter": (
                round(storage_bytes * 8 / parameter_count, 3) if parameter_count else 0.0
            ),
        }

    def unload(self) -> None:
        _json_request(
            f"{self.base_url}/api/generate",
            {"model": self.model, "keep_alive": 0},
            timeout=30,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if not _model_entry(_json_request(f"{self.base_url}/api/ps"), self.model):
                return
            time.sleep(0.1)
        raise RuntimeError(f"timed out unloading {self.model}")

    def run_case(
        self,
        case: BenchmarkCase,
        *,
        iteration: int,
        cold_start: bool,
    ) -> BenchmarkRun:
        payload = {
            "model": self.model,
            "prompt": case.prompt,
            "stream": True,
            "think": False,
            "keep_alive": "10m",
            "options": {
                "num_predict": case.max_tokens,
                "temperature": 0,
                "seed": 7,
            },
        }
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
            method="POST",
        )
        started = time.perf_counter()
        first_token_at: float | None = None
        response_parts: list[str] = []
        final: dict = {}
        with MemorySampler() as memory:
            with urlopen(request, timeout=300) as response:
                for raw_line in response:
                    event = json.loads(raw_line.decode("utf-8"))
                    text = str(event.get("response", ""))
                    if text and first_token_at is None:
                        first_token_at = time.perf_counter()
                    response_parts.append(text)
                    if event.get("done"):
                        final = event
        ended = time.perf_counter()
        resident = _model_entry(_json_request(f"{self.base_url}/api/ps"), self.model)
        model_resident = int(resident.get("size", 0))
        accelerator_resident = int(resident.get("size_vram", 0))
        return BenchmarkRun(
            case=case.name,
            iteration=iteration,
            cold_start=cold_start,
            wall_time_seconds=round(ended - started, 6),
            time_to_first_token_seconds=(
                round(first_token_at - started, 6) if first_token_at is not None else None
            ),
            prompt_tokens=int(final.get("prompt_eval_count", 0)),
            completion_tokens=int(final.get("eval_count", 0)),
            prompt_tokens_per_second=tokens_per_second(
                int(final.get("prompt_eval_count", 0)),
                int(final.get("prompt_eval_duration", 0)),
            ),
            generation_tokens_per_second=tokens_per_second(
                int(final.get("eval_count", 0)), int(final.get("eval_duration", 0))
            ),
            ollama_total_seconds=round(int(final.get("total_duration", 0)) / NANOSECONDS, 6),
            model_load_seconds=round(int(final.get("load_duration", 0)) / NANOSECONDS, 6),
            prompt_eval_seconds=round(int(final.get("prompt_eval_duration", 0)) / NANOSECONDS, 6),
            generation_seconds=round(int(final.get("eval_duration", 0)) / NANOSECONDS, 6),
            ollama_rss_before_bytes=memory.before.ollama_rss_bytes,
            ollama_rss_peak_bytes=memory.peak.ollama_rss_bytes,
            ollama_rss_after_bytes=memory.after.ollama_rss_bytes,
            system_used_before_bytes=memory.before.system_used_bytes,
            system_used_peak_bytes=memory.peak.system_used_bytes,
            system_used_after_bytes=memory.after.system_used_bytes,
            model_resident_bytes=model_resident,
            accelerator_resident_bytes=accelerator_resident,
            reported_host_offload_bytes=max(0, model_resident - accelerator_resident),
            response_characters=len("".join(response_parts)),
            response_sha256=hashlib.sha256("".join(response_parts).encode("utf-8")).hexdigest(),
        )

    def run_suite(
        self,
        *,
        cases: tuple[BenchmarkCase, ...] = DEFAULT_CASES,
        repetitions: int = 3,
        warmups: int = 1,
    ) -> dict:
        if repetitions < 1 or warmups < 0 or not cases:
            raise ValueError("benchmark requires cases, repetitions >= 1, and warmups >= 0")
        metadata = self.model_metadata()
        if metadata["format"] != "gguf":
            raise RuntimeError(f"{self.model} is not GGUF: {metadata['format']}")
        disk_before = shutil.disk_usage(Path.home())
        runs: list[BenchmarkRun] = []
        self.unload()
        runs.append(self.run_case(cases[0], iteration=0, cold_start=True))
        for warmup in range(warmups):
            self.run_case(cases[warmup % len(cases)], iteration=-(warmup + 1), cold_start=False)
        for case in cases:
            for iteration in range(1, repetitions + 1):
                runs.append(self.run_case(case, iteration=iteration, cold_start=False))
        disk_after = shutil.disk_usage(Path.home())
        last_run = runs[-1]
        return {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "system_memory_bytes": int(
                    subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
                ),
            },
            "engine": {
                "name": "ollama",
                "base_url": self.base_url,
                "version": subprocess.check_output(["ollama", "--version"], text=True).strip(),
                "num_gpu": self.num_gpu,
            },
            "model": metadata,
            "residency": {
                "model_resident_bytes": last_run.model_resident_bytes,
                "accelerator_resident_bytes": last_run.accelerator_resident_bytes,
                "reported_host_offload_bytes": last_run.reported_host_offload_bytes,
                "accelerator_residency_percent": (
                    round(
                        100 * last_run.accelerator_resident_bytes / last_run.model_resident_bytes,
                        3,
                    )
                    if last_run.model_resident_bytes
                    else 0.0
                ),
            },
            "benchmark": {
                "warmups": warmups,
                "repetitions": repetitions,
                "cases": [asdict(case) for case in cases],
                "runs": [asdict(run) for run in runs],
                "summary": summarize_runs(runs),
                "cold_start": asdict(runs[0]),
            },
            "storage": {
                "disk_free_before_bytes": disk_before.free,
                "disk_free_after_bytes": disk_after.free,
                "disk_delta_bytes": disk_after.free - disk_before.free,
            },
            "metric_notes": {
                "system_used_bytes": (
                    "macOS active + wired + compressor pages; it is not total allocated memory"
                ),
                "reported_host_offload_bytes": (
                    "Ollama reported model size minus size_vram; Apple unified memory is shared"
                ),
                "ollama_rss_bytes": (
                    "RSS for visible Ollama serve/runner processes; Metal allocations are "
                    "primarily represented by residency and system-used metrics"
                ),
                "effective_storage_bits_per_parameter": (
                    "stored model bytes divided by parameter count; includes container metadata"
                ),
            },
        }
