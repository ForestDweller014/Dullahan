from __future__ import annotations

import argparse
from pathlib import Path

from graph_builder.cluster import generate_clusters
from graph_builder.experts import generate_experts_from_clusters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate K-sized graph clusters.")
    parser.add_argument("--graph-dir", type=Path, default=Path("memory/graph"))
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--cluster-prefix", default="cluster:auto")
    parser.add_argument(
        "--write-experts",
        action="store_true",
        help="Regenerate experts.yaml and missing role-context Markdown docs from clusters.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate_clusters(
        graph_dir=args.graph_dir,
        k=args.k,
        cluster_prefix=args.cluster_prefix,
    )
    if args.write_experts:
        generate_experts_from_clusters(graph_dir=args.graph_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
