from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dullahan_kg.graph import KnowledgeGraph
from dullahan_kg.storage.yaml_graph_store import YamlGraphStore
from dullahan_shared.embeddings import EmbeddingModel
from dullahan_shared.schemas.graph import EdgeType, GraphEdge, GraphNode, NodeType
from world_state import LocalWorldStateDB

from graph_builder.cluster import generate_clusters
from graph_builder.experts import generate_experts_from_clusters


@dataclass(frozen=True)
class GraphifyConfig:
    source_path: Path
    repo_root: Path
    graph_dir: Path
    documents_dir: Path
    k: int
    graphify_command: str = "graphify"
    graphify_output_dir: Path | None = None


def graphify_collection(
    config: GraphifyConfig,
    *,
    embedding_model: EmbeddingModel | None = None,
) -> KnowledgeGraph:
    graphify_json_path = run_graphify(config)
    return import_graphify_json(
        config=config,
        graphify_json_path=graphify_json_path,
        embedding_model=embedding_model,
    )


def import_graphify_json(
    *,
    config: GraphifyConfig,
    graphify_json_path: Path,
    embedding_model: EmbeddingModel | None = None,
) -> KnowledgeGraph:
    payload = _read_graphify_json(graphify_json_path)
    graph = _convert_graphify_payload(config=config, payload=payload)
    YamlGraphStore(config.graph_dir).save(graph)
    generate_clusters(graph_dir=config.graph_dir, k=config.k)
    generate_experts_from_clusters(
        graph_dir=config.graph_dir,
        repo_root=config.repo_root,
    )
    LocalWorldStateDB.from_graph_memory(
        repo_root=config.repo_root,
        graph_dir=config.graph_dir,
        embedding_model=embedding_model,
    ).rebuild_index()
    return YamlGraphStore(config.graph_dir).load()


def run_graphify(config: GraphifyConfig) -> Path:
    source_path = config.source_path.resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"collection path does not exist: {source_path}")
    if shutil.which(config.graphify_command) is None:
        raise RuntimeError(
            f"could not find the `{config.graphify_command}` command. "
            "Install safishamsi/graphify with `python -m pip install graphifyy`."
        )

    output_dir = config.graphify_output_dir or config.repo_root / "graphify-out"
    subprocess.run(
        [config.graphify_command, source_path.as_posix()],
        cwd=config.repo_root,
        check=True,
    )

    graphify_json_path = output_dir / "graph.json"
    if not graphify_json_path.exists():
        raise FileNotFoundError(
            "graphify completed, but Dullahan could not find "
            f"{graphify_json_path}. If graphify wrote elsewhere, pass "
            "`--graphify-output-dir`."
        )
    return graphify_json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run graphify and import its graph into Dullahan graph memory.",
    )
    parser.add_argument(
        "collection",
        type=Path,
        nargs="?",
        help="File or directory to pass to graphify.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--graph-dir", type=Path, default=Path("memory/graph"))
    parser.add_argument("--documents-dir", type=Path, default=Path("memory/documents/nodes"))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--graphify-command", default="graphify")
    parser.add_argument("--graphify-output-dir", type=Path, default=None)
    parser.add_argument(
        "--from-graphify-json",
        type=Path,
        default=None,
        help="Skip running graphify and import an existing graphify graph.json.",
    )
    parser.add_argument(
        "--postgres-dsn",
        default=None,
        help="PostgreSQL DSN to pull source context before graphification.",
    )
    parser.add_argument(
        "--postgres-query",
        default=(
            "SELECT id, title, content, metadata "
            "FROM dullahan_context_documents "
            "ORDER BY id"
        ),
        help=(
            "SQL query used with --postgres-dsn. It should return id, title, content, "
            "and optionally metadata columns."
        ),
    )
    parser.add_argument(
        "--postgres-export-dir",
        type=Path,
        default=Path("memory/postgres_context"),
        help="Directory where pulled PostgreSQL rows are written before Graphify runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    graph_dir = _resolve_under_repo(repo_root, args.graph_dir)
    documents_dir = _resolve_under_repo(repo_root, args.documents_dir)
    output_dir = (
        _resolve_under_repo(repo_root, args.graphify_output_dir)
        if args.graphify_output_dir is not None
        else None
    )
    source_path = args.collection
    if args.postgres_dsn is not None:
        source_path = export_postgres_context(
            dsn=args.postgres_dsn,
            query=args.postgres_query,
            export_dir=_resolve_under_repo(repo_root, args.postgres_export_dir),
        )
    if source_path is None and args.from_graphify_json is None:
        raise SystemExit(
            "collection is required unless --postgres-dsn or --from-graphify-json is set"
        )

    config = GraphifyConfig(
        source_path=source_path or repo_root,
        repo_root=repo_root,
        graph_dir=graph_dir,
        documents_dir=documents_dir,
        k=args.k,
        graphify_command=args.graphify_command,
        graphify_output_dir=output_dir,
    )
    graph = (
        import_graphify_json(config=config, graphify_json_path=args.from_graphify_json)
        if args.from_graphify_json is not None
        else graphify_collection(config)
    )
    print(
        "Imported graphify graph: "
        f"{len(graph.nodes)} nodes, {len(graph.edges)} edges, {len(graph.clusters)} clusters"
    )
    print(f"Graph memory: {graph_dir}")
    return 0


def export_postgres_context(*, dsn: str, query: str, export_dir: Path) -> Path:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL graphification requires psycopg. "
            "Install project dependencies with `python -m pip install -e \".[dev]\"`."
        ) from exc

    export_dir.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

    for stale_file in export_dir.glob("*.md"):
        stale_file.unlink()

    for index, row in enumerate(rows):
        document_id = str(row.get("id") or f"postgres-row-{index}")
        title = str(row.get("title") or document_id)
        content = str(row.get("content") or row.get("text") or "")
        metadata = row.get("metadata") or {}
        path = export_dir / f"{_slug(document_id)}.md"
        path.write_text(
            "\n".join(
                [
                    f"# {title}",
                    "",
                    f"- PostgreSQL ID: `{document_id}`",
                    "",
                    "## Metadata",
                    "",
                    "```json",
                    json.dumps(metadata, indent=2, sort_keys=True, default=str),
                    "```",
                    "",
                    "## Content",
                    "",
                    content,
                    "",
                ]
            ),
            encoding="utf-8",
        )

    return export_dir


def _read_graphify_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected graphify JSON object in {path}")
    return payload


def _convert_graphify_payload(
    *,
    config: GraphifyConfig,
    payload: dict[str, Any],
) -> KnowledgeGraph:
    raw_nodes = _as_list(payload.get("nodes", []))
    raw_edges = _as_list(payload.get("edges", []))
    nodes = [
        _convert_node(config=config, raw_node=raw_node, fallback_index=index)
        for index, raw_node in enumerate(raw_nodes)
        if isinstance(raw_node, dict)
    ]
    node_ids = {node.id for node in nodes}
    edges = [
        edge
        for index, raw_edge in enumerate(raw_edges)
        if isinstance(raw_edge, dict)
        for edge in [_convert_edge(raw_edge, fallback_index=index, node_ids=node_ids)]
        if edge is not None
    ]
    return KnowledgeGraph.from_parts(nodes=nodes, edges=_dedupe_edges(edges))


def _convert_node(
    *,
    config: GraphifyConfig,
    raw_node: dict[str, Any],
    fallback_index: int,
) -> GraphNode:
    raw_id = str(raw_node.get("id") or raw_node.get("name") or f"node:{fallback_index}")
    title = str(raw_node.get("label") or raw_node.get("name") or raw_node.get("title") or raw_id)
    node_id = _node_id(raw_id)
    document_path = _write_node_document(
        config=config,
        node_id=node_id,
        title=title,
        raw_node=raw_node,
    )
    return GraphNode(
        id=node_id,
        type=_node_type(raw_node),
        title=title,
        document_paths=[document_path],
        metadata={
            "graphify_id": raw_id,
            "graphify_type": str(raw_node.get("type") or raw_node.get("kind") or ""),
            "graphify_source_file": str(raw_node.get("source_file") or ""),
        },
    )


def _convert_edge(
    raw_edge: dict[str, Any],
    *,
    fallback_index: int,
    node_ids: set[str],
) -> GraphEdge | None:
    source = raw_edge.get("source") or raw_edge.get("source_id") or raw_edge.get("from")
    target = raw_edge.get("target") or raw_edge.get("target_id") or raw_edge.get("to")
    if source is None or target is None:
        return None
    source_id = _node_id(str(source))
    target_id = _node_id(str(target))
    if source_id not in node_ids or target_id not in node_ids:
        return None

    relation = str(
        raw_edge.get("relation")
        or raw_edge.get("type")
        or raw_edge.get("label")
        or "semantically_related"
    )
    return GraphEdge(
        source_id=source_id,
        target_id=target_id,
        type=_edge_type(relation),
        weight=_edge_weight(raw_edge),
        metadata={
            "graphify_relation": relation,
            "graphify_edge_index": fallback_index,
        },
    )


def _write_node_document(
    *,
    config: GraphifyConfig,
    node_id: str,
    title: str,
    raw_node: dict[str, Any],
) -> str:
    config.documents_dir.mkdir(parents=True, exist_ok=True)
    path = config.documents_dir / f"{_slug(node_id)}.md"
    text = raw_node.get("description") or raw_node.get("content") or raw_node.get("summary") or ""
    content = "\n".join(
        [
            f"# {title}",
            "",
            f"- Node ID: `{node_id}`",
            f"- Graphify ID: `{raw_node.get('id', '')}`",
            f"- Source file: `{raw_node.get('source_file', '')}`",
            "",
            "## Graphify Metadata",
            "",
            "```json",
            json.dumps(raw_node, indent=2, sort_keys=True),
            "```",
            "",
            "## Content",
            "",
            str(text),
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path.relative_to(config.repo_root).as_posix()


def _node_type(raw_node: dict[str, Any]) -> NodeType:
    raw_type = str(raw_node.get("type") or raw_node.get("kind") or "").lower()
    file_type = str(raw_node.get("file_type") or raw_node.get("source_file") or "").lower()
    if "class" in raw_type:
        return NodeType.CLASS
    if "service" in raw_type:
        return NodeType.SERVICE
    if "image" in raw_type or file_type.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        return NodeType.IMAGE
    if raw_node.get("source_file") or "file" in raw_type:
        return NodeType.FILE
    return NodeType.CONCEPT


def _edge_type(relation: str) -> EdgeType:
    normalized = relation.lower()
    if "import" in normalized:
        return EdgeType.IMPORTS
    if "own" in normalized or "contain" in normalized:
        return EdgeType.OWNS
    if "reference" in normalized or "refer" in normalized:
        return EdgeType.REFERENCES
    if "connect" in normalized or "call" in normalized:
        return EdgeType.CONNECTS_TO
    return EdgeType.SEMANTICALLY_RELATED


def _edge_weight(raw_edge: dict[str, Any]) -> float:
    for key in ("weight", "confidence", "score"):
        value = raw_edge.get(key)
        if isinstance(value, int | float):
            return max(float(value), 0.0)
    return 1.0


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return []


def _dedupe_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    deduped: dict[tuple[str, str, EdgeType], GraphEdge] = {}
    for edge in edges:
        key = (edge.source_id, edge.target_id, edge.type)
        existing = deduped.get(key)
        if existing is None or edge.weight > existing.weight:
            deduped[key] = edge
    return list(deduped.values())


def _node_id(graphify_id: str) -> str:
    return f"concept:graphify:{_slug(graphify_id)}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return slug or "node"


def _resolve_under_repo(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
