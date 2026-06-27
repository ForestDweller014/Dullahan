from __future__ import annotations

from collections import defaultdict

from dullahan_kg.graph import GraphCluster, KnowledgeGraph


def partition_by_k(graph: KnowledgeGraph, *, k: int, cluster_prefix: str = "cluster:auto") -> list[GraphCluster]:
    """Partition graph nodes into deterministic topology-aware clusters of size at most K."""
    if k <= 0:
        raise ValueError("k must be positive")

    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for edge in graph.edges:
        adjacency[edge.source_id][edge.target_id] = max(
            edge.weight,
            adjacency[edge.source_id].get(edge.target_id, 0.0),
        )
        adjacency[edge.target_id][edge.source_id] = max(
            edge.weight,
            adjacency[edge.target_id].get(edge.source_id, 0.0),
        )

    unassigned = set(graph.nodes)
    clusters: list[GraphCluster] = []

    while unassigned:
        seed_id = _choose_seed(unassigned, adjacency)
        node_ids = [seed_id]
        unassigned.remove(seed_id)

        while unassigned and len(node_ids) < k:
            next_id = _choose_next(node_ids, unassigned, adjacency)
            node_ids.append(next_id)
            unassigned.remove(next_id)

        cluster_number = len(clusters) + 1
        clusters.append(
            GraphCluster(
                id=f"{cluster_prefix}:{cluster_number}",
                title=f"Auto Cluster {cluster_number}",
                node_ids=node_ids,
                metadata={
                    "generated_by": "partition_by_k",
                    "k": k,
                    "seed_node_id": seed_id,
                },
            )
        )

    return clusters


def _choose_seed(unassigned: set[str], adjacency: dict[str, dict[str, float]]) -> str:
    return sorted(
        unassigned,
        key=lambda node_id: (
            -sum(adjacency[node_id].values()),
            -len(adjacency[node_id]),
            node_id,
        ),
    )[0]


def _choose_next(
    cluster_node_ids: list[str],
    unassigned: set[str],
    adjacency: dict[str, dict[str, float]],
) -> str:
    return sorted(
        unassigned,
        key=lambda node_id: (
            -sum(adjacency[node_id].get(cluster_node_id, 0.0) for cluster_node_id in cluster_node_ids),
            -sum(adjacency[node_id].values()),
            node_id,
        ),
    )[0]
