import pytest

from dullahan_kg.algorithms import partition_by_k
from dullahan_kg.graph import KnowledgeGraph
from dullahan_shared.schemas.graph import EdgeType, GraphEdge, GraphNode, NodeType


def test_partition_by_k_limits_cluster_size_and_assigns_every_node() -> None:
    nodes = [
        GraphNode(id=f"concept:{name}", type=NodeType.CONCEPT, title=name)
        for name in ["a", "b", "c", "d", "e"]
    ]
    edges = [
        GraphEdge(source_id="concept:a", target_id="concept:b", type=EdgeType.CONNECTS_TO, weight=1.0),
        GraphEdge(source_id="concept:b", target_id="concept:c", type=EdgeType.CONNECTS_TO, weight=1.0),
        GraphEdge(source_id="concept:d", target_id="concept:e", type=EdgeType.CONNECTS_TO, weight=1.0),
    ]
    graph = KnowledgeGraph.from_parts(nodes=nodes, edges=edges)

    clusters = partition_by_k(graph, k=2)
    assigned = [node_id for cluster in clusters for node_id in cluster.node_ids]

    assert all(len(cluster.node_ids) <= 2 for cluster in clusters)
    assert sorted(assigned) == sorted(graph.nodes)
    assert clusters[0].metadata["generated_by"] == "partition_by_k"


def test_partition_by_k_rejects_non_positive_k() -> None:
    graph = KnowledgeGraph.from_parts(nodes=[], edges=[])

    with pytest.raises(ValueError, match="k must be positive"):
        partition_by_k(graph, k=0)
