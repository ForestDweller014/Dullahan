from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import uvicorn

from dullahan_inference.api import create_ollama_app
from dullahan_inference.config import InferenceConfig, InferenceProvider
from dullahan_inference.model_server import (
    ModelServerError,
    activate_model_server,
    export_model_server_package,
    get_model_server_metadata,
)
from dullahan_inference.plan import resolve_inference_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local Qwen/vLLM or Ollama inference.")
    parser.add_argument(
        "command",
        choices=("activate", "export", "metadata", "plan", "serve"),
        nargs="?",
        default="serve",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="destination archive for the export command",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.getenv("DULLAHAN_INFERENCE_CONFIG", "configs/inference.yaml")),
    )
    return parser


def run_from_args(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = InferenceConfig.from_yaml(args.config)
    plan = resolve_inference_plan(config)
    if args.command == "plan":
        print(json.dumps(plan.model_dump(mode="json"), indent=2))
        return 0

    if args.command == "activate":
        if not config.model_server.enabled:
            raise SystemExit("activate requires model_server.enabled=true")
        try:
            result = activate_model_server(plan)
        except ModelServerError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(result, indent=2))
        return 0

    if args.command in {"metadata", "export"}:
        if not config.model_server.enabled:
            raise SystemExit(f"{args.command} requires model_server.enabled=true")
        try:
            if args.command == "metadata":
                print(json.dumps(get_model_server_metadata(plan), indent=2))
            else:
                if args.output is None:
                    raise SystemExit("export requires --output")
                print(export_model_server_package(plan, args.output))
        except ModelServerError as exc:
            raise SystemExit(str(exc)) from exc
        return 0

    if not plan.memory_fit:
        raise SystemExit(
            "inference plan does not fit available memory; inspect "
            "`dullahan-inference plan` and adjust model, offload, or memory reserves"
        )

    if config.model_server.enabled:
        raise SystemExit(
            "the container model server is managed with Docker Compose; use the "
            "model-server README to start it, then run `dullahan-inference activate`"
        )

    if config.provider == InferenceProvider.QWEN:
        if shutil.which(config.vllm.executable) is None:
            raise SystemExit(
                f"could not find {config.vllm.executable!r}; install a platform-appropriate "
                "vLLM build before serving Qwen"
            )
        environment = os.environ.copy()
        environment.update(plan.environment)
        os.execvpe(plan.command[0], plan.command, environment)

    app = create_ollama_app(config, plan)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
    )
    return 0


def main() -> int:
    return run_from_args()


if __name__ == "__main__":
    raise SystemExit(main())
