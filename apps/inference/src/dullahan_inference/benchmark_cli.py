from __future__ import annotations

import argparse
import json
from pathlib import Path

from dullahan_inference.benchmark import BenchmarkCase, OllamaGGUFBenchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark local Qwen GGUF inference.")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--num-gpu",
        type=int,
        help="Ollama GPU-layer override; use 0 for a CPU-placement comparison",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run one 48-token case for placement and smoke comparisons",
    )
    args = parser.parse_args(argv)

    cases = (
        (
            BenchmarkCase(
                name="quick_placement",
                prompt="Explain in two sentences where this model is executing.",
                max_tokens=48,
            ),
        )
        if args.quick
        else None
    )

    benchmark = OllamaGGUFBenchmark(
        model=args.model,
        base_url=args.base_url,
        num_gpu=args.num_gpu,
    )
    run_options = {"repetitions": args.repetitions, "warmups": args.warmups}
    if cases is not None:
        run_options["cases"] = cases
    report = benchmark.run_suite(**run_options)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0
