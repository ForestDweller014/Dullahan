from __future__ import annotations

import os

import pytest
from dullahan_inference.benchmark import BenchmarkCase, OllamaGGUFBenchmark


@pytest.mark.local_inference
def test_local_qwen_gguf_benchmark_records_performance_and_memory() -> None:
    if os.getenv("DULLAHAN_RUN_LOCAL_INFERENCE") != "1":
        pytest.skip("set DULLAHAN_RUN_LOCAL_INFERENCE=1 to benchmark local Qwen GGUF")

    report = OllamaGGUFBenchmark(model=os.getenv("DULLAHAN_TEST_MODEL", "qwen3:8b")).run_suite(
        cases=(
            BenchmarkCase(
                name="pytest_smoke",
                prompt="Explain GGUF quantization in three concise sentences.",
                max_tokens=96,
            ),
        ),
        repetitions=1,
        warmups=0,
    )

    model = report["model"]
    cold = report["benchmark"]["cold_start"]
    warm = report["benchmark"]["runs"][1]
    assert model["format"] == "gguf"
    assert str(model["quantization_level"]).startswith("Q4")
    assert model["parameter_count"] >= 7_000_000_000
    assert 4 <= model["effective_storage_bits_per_parameter"] < 8
    assert cold["model_load_seconds"] > 0
    assert cold["model_load_seconds"] < 15
    assert cold["time_to_first_token_seconds"] is not None
    assert cold["time_to_first_token_seconds"] < 20
    assert warm["completion_tokens"] > 0
    assert warm["generation_tokens_per_second"] > 10
    assert warm["time_to_first_token_seconds"] < 3
    assert warm["ollama_rss_peak_bytes"] > 0
    assert warm["model_resident_bytes"] >= warm["accelerator_resident_bytes"] > 0
    assert warm["reported_host_offload_bytes"] == (
        warm["model_resident_bytes"] - warm["accelerator_resident_bytes"]
    )
    assert cold["system_used_peak_bytes"] - cold["system_used_before_bytes"] < 10 * 1024**3
