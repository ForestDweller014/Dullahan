from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dullahan_kg.graph import GraphCluster, KnowledgeGraph
from dullahan_kg.storage.yaml_graph_store import YamlGraphStore


def generate_experts_from_clusters(
    *,
    graph_dir: Path,
    repo_root: Path | None = None,
    experts_path: Path | None = None,
    role_context_dir: Path | None = None,
    default_model_prefix: str = "local-slm",
    default_max_concurrency: int = 4,
) -> None:
    store = YamlGraphStore(graph_dir)
    graph = store.load()
    resolved_repo_root = repo_root or _infer_repo_root(graph_dir)
    resolved_experts_path = experts_path or graph_dir / "experts.yaml"
    resolved_role_context_dir = role_context_dir or resolved_repo_root / "memory" / "documents" / "clusters"

    resolved_role_context_dir.mkdir(parents=True, exist_ok=True)
    experts = [
        _expert_for_cluster(
            graph=graph,
            cluster=cluster,
            repo_root=resolved_repo_root,
            role_context_dir=resolved_role_context_dir,
            default_model_prefix=default_model_prefix,
            default_max_concurrency=default_max_concurrency,
        )
        for cluster in graph.clusters.values()
    ]

    resolved_experts_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_experts_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump({"experts": experts}, stream, sort_keys=False)


def _expert_for_cluster(
    *,
    graph: KnowledgeGraph,
    cluster: GraphCluster,
    repo_root: Path,
    role_context_dir: Path,
    default_model_prefix: str,
    default_max_concurrency: int,
) -> dict[str, Any]:
    role_context_path = _ensure_role_context_document(
        graph=graph,
        cluster=cluster,
        repo_root=repo_root,
        role_context_dir=role_context_dir,
    )
    slug = _slug(cluster.id)
    return {
        "id": f"expert:{slug}",
        "cluster_id": cluster.id,
        "role_context_path": role_context_path,
        "model": f"{default_model_prefix}-{slug}",
        "max_concurrency": default_max_concurrency,
        "metadata": {
            "generated_by": "graph_builder.generate_experts_from_clusters",
            "cluster_title": cluster.title,
            "node_count": len(cluster.node_ids),
        },
    }


def _ensure_role_context_document(
    *,
    graph: KnowledgeGraph,
    cluster: GraphCluster,
    repo_root: Path,
    role_context_dir: Path,
) -> str:
    if cluster.document_paths:
        return cluster.document_paths[0]

    path = role_context_dir / f"{_slug(cluster.id)}.md"
    node_lines = []
    for node_id in cluster.node_ids:
        node = graph.get_node(node_id)
        node_lines.append(f"- `{node.id}` ({node.type}): {node.title}")
        for document_path in node.document_paths:
            node_lines.append(f"  - document: `{document_path}`")

    text = "\n".join(
        [
            f"# {cluster.title}",
            "",
            f"Expert role context generated for `{cluster.id}`.",
            "",
            "This expert is responsible for answering subqueries whose semantics",
            "route to this knowledge-graph cluster.",
            "",
            "## Nodes",
            "",
            *node_lines,
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path.relative_to(repo_root).as_posix()


def _infer_repo_root(graph_dir: Path) -> Path:
    if graph_dir.name == "graph" and graph_dir.parent.name == "memory":
        return graph_dir.parent.parent
    return graph_dir.parent


def _slug(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")
