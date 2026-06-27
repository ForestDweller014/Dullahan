from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from pydantic import BaseModel, Field

from dullahan_shared.schemas.graph import GraphEdge, GraphNode


class GraphCluster(BaseModel):
    id: str
    title: str
    node_ids: list[str] = Field(default_factory=list)
    document_paths: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class KnowledgeGraph(BaseModel):
    nodes: dict[str, GraphNode] = Field(default_factory=dict)
    edges: list[GraphEdge] = Field(default_factory=list)
    clusters: dict[str, GraphCluster] = Field(default_factory=dict)

    @classmethod
    def from_parts(
        cls,
        nodes: Iterable[GraphNode],
        edges: Iterable[GraphEdge],
        clusters: Iterable[GraphCluster] = (),
    ) -> KnowledgeGraph:
        graph = cls(
            nodes={node.id: node for node in nodes},
            edges=list(edges),
            clusters={cluster.id: cluster for cluster in clusters},
        )
        graph.validate_references()
        return graph

    def validate_references(self) -> None:
        missing_edge_refs: list[str] = []
        for edge in self.edges:
            if edge.source_id not in self.nodes:
                missing_edge_refs.append(edge.source_id)
            if edge.target_id not in self.nodes:
                missing_edge_refs.append(edge.target_id)

        missing_cluster_refs: list[str] = []
        for cluster in self.clusters.values():
            missing_cluster_refs.extend(node_id for node_id in cluster.node_ids if node_id not in self.nodes)

        if missing_edge_refs or missing_cluster_refs:
            details = []
            if missing_edge_refs:
                details.append(f"edge references: {sorted(set(missing_edge_refs))}")
            if missing_cluster_refs:
                details.append(f"cluster references: {sorted(set(missing_cluster_refs))}")
            raise ValueError(f"knowledge graph contains missing node references ({'; '.join(details)})")

    def get_node(self, node_id: str) -> GraphNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown graph node: {node_id}") from exc

    def outgoing_edges(self, node_id: str) -> list[GraphEdge]:
        return [edge for edge in self.edges if edge.source_id == node_id]

    def incoming_edges(self, node_id: str) -> list[GraphEdge]:
        return [edge for edge in self.edges if edge.target_id == node_id]

    def neighbors(self, node_id: str, depth: int = 1) -> list[GraphNode]:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.get_node(node_id)

        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)

        visited = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        ordered: list[GraphNode] = []

        while queue:
            current_id, current_depth = queue.popleft()
            if current_depth == depth:
                continue

            for neighbor_id in sorted(adjacency[current_id]):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                ordered.append(self.nodes[neighbor_id])
                queue.append((neighbor_id, current_depth + 1))

        return ordered

    def nodes_for_cluster(self, cluster_id: str) -> list[GraphNode]:
        try:
            cluster = self.clusters[cluster_id]
        except KeyError as exc:
            raise KeyError(f"unknown graph cluster: {cluster_id}") from exc
        return [self.get_node(node_id) for node_id in cluster.node_ids]
