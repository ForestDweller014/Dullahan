from __future__ import annotations

from dullahan_inference.benchmark import (
    BenchmarkRun,
    OllamaGGUFBenchmark,
    percentile,
    summarize_runs,
    tokens_per_second,
)


def run(*, latency: float, throughput: float, cold: bool = False) -> BenchmarkRun:
    return BenchmarkRun(
        case="test",
        iteration=0,
        cold_start=cold,
        wall_time_seconds=latency,
        time_to_first_token_seconds=latency / 2,
        prompt_tokens=10,
        completion_tokens=20,
        prompt_tokens_per_second=100,
        generation_tokens_per_second=throughput,
        ollama_total_seconds=latency,
        model_load_seconds=0,
        prompt_eval_seconds=0.1,
        generation_seconds=0.2,
        ollama_rss_before_bytes=100,
        ollama_rss_peak_bytes=300,
        ollama_rss_after_bytes=250,
        system_used_before_bytes=1_000,
        system_used_peak_bytes=2_000,
        system_used_after_bytes=1_500,
        model_resident_bytes=900,
        accelerator_resident_bytes=700,
        reported_host_offload_bytes=200,
        response_characters=80,
        response_sha256=f"hash-{latency}",
    )


def test_tokens_per_second_uses_native_nanosecond_duration() -> None:
    assert tokens_per_second(25, 500_000_000) == 50
    assert tokens_per_second(25, 0) == 0


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([1, 2, 3, 4], 0.95) == 4


def test_summary_excludes_cold_start_from_warm_metrics() -> None:
    summary = summarize_runs(
        [run(latency=20, throughput=1, cold=True), run(latency=2, throughput=10)]
    )

    assert summary["measured_runs"] == 1
    assert summary["median_wall_time_seconds"] == 2
    assert summary["median_generation_tokens_per_second"] == 10
    assert summary["peak_system_used_increase_bytes"] == 1_000


def test_model_metadata_reports_effective_quantization_density(monkeypatch) -> None:
    def fake_request(url, payload=None, timeout=120):
        if url.endswith("/api/show"):
            return {
                "details": {
                    "format": "gguf",
                    "family": "qwen3",
                    "parameter_size": "8.2B",
                    "quantization_level": "Q4_K_M",
                },
                "model_info": {
                    "general.parameter_count": 8_000_000_000,
                    "general.quantization_version": 2,
                },
            }
        return {"models": [{"name": "qwen3:8b", "size": 5_000_000_000}]}

    monkeypatch.setattr("dullahan_inference.benchmark._json_request", fake_request)

    metadata = OllamaGGUFBenchmark(model="qwen3:8b").model_metadata()

    assert metadata["format"] == "gguf"
    assert metadata["quantization_level"] == "Q4_K_M"
    assert metadata["effective_storage_bits_per_parameter"] == 5.0
