from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dullahan_kg.graph import GraphCluster, KnowledgeGraph
from dullahan_shared.schemas.graph import GraphEdge, GraphNode


class YamlGraphStore:
    """Load and save a knowledge graph from a small YAML file set."""

    def __init__(self, graph_dir: Path | str) -> None:
        self.graph_dir = Path(graph_dir)

    @property
    def graph_path(self) -> Path:
        return self.graph_dir / "graph.yaml"

    @property
    def clusters_path(self) -> Path:
        return self.graph_dir / "clusters.yaml"

    def load(self) -> KnowledgeGraph:
        graph_data = self._read_yaml(self.graph_path, default={"nodes": [], "edges": []})
        cluster_data = self._read_yaml(self.clusters_path, default={"clusters": []})

        nodes = [GraphNode.model_validate(item) for item in graph_data.get("nodes", [])]
        edges = [GraphEdge.model_validate(item) for item in graph_data.get("edges", [])]
        clusters = [GraphCluster.model_validate(item) for item in cluster_data.get("clusters", [])]

        return KnowledgeGraph.from_parts(nodes=nodes, edges=edges, clusters=clusters)

    def save(self, graph: KnowledgeGraph) -> None:
        graph.validate_references()
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self._write_yaml(
            self.graph_path,
            {
                "nodes": [node.model_dump(mode="json") for node in graph.nodes.values()],
                "edges": [edge.model_dump(mode="json") for edge in graph.edges],
            },
        )
        self._write_yaml(
            self.clusters_path,
            {
                "clusters": [
                    cluster.model_dump(mode="json") for cluster in graph.clusters.values()
                ],
            },
        )

    def _read_yaml(self, path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or default
        if not isinstance(data, dict):
            raise ValueError(f"expected YAML mapping in {path}")
        return data

    def _write_yaml(self, path: Path, data: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
