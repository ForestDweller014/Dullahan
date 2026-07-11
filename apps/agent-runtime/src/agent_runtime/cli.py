from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_runtime.agent import AgentRuntime
from agent_runtime.config import AgentRuntimeConfig
from agent_runtime.models import AgentRunRequest, AgentRunResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Dullahan agent runtime.")
    parser.add_argument("query", help="Root query to execute.")
    parser.add_argument(
        "--repo-root",
        default=Path.cwd(),
        type=Path,
        help="Repository root containing configs and memory.",
    )
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-breadth", type=int, default=None)
    parser.add_argument("--max-total-instances", type=int, default=None)
    parser.add_argument(
        "--transport",
        choices=["local", "http"],
        default="local",
        help="Run CAL/EDL in-process or call remote HTTP services.",
    )
    parser.add_argument("--cal-url", default="http://127.0.0.1:8100")
    parser.add_argument("--edl-url", default="http://127.0.0.1:8200")
    parser.add_argument("--tool-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--persist-artifacts",
        action="store_true",
        help="Write YAML/Markdown execution artifacts under memory/executions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full run result as JSON.",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> AgentRunResult:
    config = AgentRuntimeConfig.from_files(args.repo_root)
    limits = config.limits.model_copy(
        update={
            key: value
            for key, value in {
                "max_depth": args.max_depth,
                "max_breadth_per_agent": args.max_breadth,
                "max_total_agent_instances": args.max_total_instances,
            }.items()
            if value is not None
        }
    )
    config = config.model_copy(update={"limits": limits})
    if args.transport == "local":
        runtime = AgentRuntime.local(config)
    else:
        runtime = AgentRuntime.remote(
            config=config,
            cal_base_url=args.cal_url,
            edl_base_url=args.edl_url,
            timeout_seconds=args.tool_timeout_seconds,
        )
    return runtime.run(
        AgentRunRequest(
            query=args.query,
            persist_artifacts=args.persist_artifacts,
        )
    )


def format_text(result: AgentRunResult) -> str:
    lines = [
        result.final_response,
        "",
        f"Trace: {result.trace_id}",
        f"Subqueries: {len(result.subqueries)}",
        f"Expert responses: {len(result.expert_responses)}",
        f"Spans: {len(result.spans)}",
    ]
    if result.artifact_dir:
        lines.append(f"Artifacts: {result.artifact_dir}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_from_args(args)

    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
    else:
        print(format_text(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
