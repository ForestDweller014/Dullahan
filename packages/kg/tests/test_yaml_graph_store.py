from pathlib import Path

import pytest

from dullahan_kg.graph import GraphCluster, KnowledgeGraph
from dullahan_kg.storage.yaml_graph_store import YamlGraphStore
from dullahan_shared.schemas.graph import EdgeType, GraphEdge, GraphNode, NodeType


def build_graph() -> KnowledgeGraph:
    cal = GraphNode(
        id="concept:cal",
        type=NodeType.CONCEPT,
        title="Context Augmentation Layer",
        document_paths=["memory/documents/nodes/concept__cal.md"],
        cluster_id="cluster:context",
    )
    edl = GraphNode(
        id="concept:edl",
        type=NodeType.CONCEPT,
        title="Expert Dispatch Layer",
        document_paths=["memory/documents/nodes/concept__edl.md"],
        cluster_id="cluster:dispatch",
    )
    edge = GraphEdge(
        source_id=cal.id,
        target_id=edl.id,
        type=EdgeType.CONNECTS_TO,
        weight=0.9,
    )
    context_cluster = GraphCluster(
        id="cluster:context",
        title="Context Memory",
        node_ids=[cal.id],
        document_paths=["memory/documents/clusters/cluster__context.md"],
    )
    dispatch_cluster = GraphCluster(
        id="cluster:dispatch",
        title="Expert Dispatch",
        node_ids=[edl.id],
        document_paths=["memory/documents/clusters/cluster__dispatch.md"],
    )
    return KnowledgeGraph.from_parts(
        nodes=[cal, edl],
        edges=[edge],
        clusters=[context_cluster, dispatch_cluster],
    )


# Verifies that YAML graph store round trips graph.
def test_yaml_graph_store_round_trips_graph(tmp_path: Path) -> None:
    store = YamlGraphStore(tmp_path / "graph")
    store.save(build_graph())

    loaded = store.load()

    assert loaded.get_node("concept:cal").title == "Context Augmentation Layer"
    assert loaded.outgoing_edges("concept:cal")[0].target_id == "concept:edl"
    assert loaded.nodes_for_cluster("cluster:dispatch")[0].id == "concept:edl"


# Verifies that neighbors walk edges without returning origin.
def test_neighbors_walk_edges_without_returning_origin() -> None:
    graph = build_graph()

    neighbors = graph.neighbors("concept:cal")

    assert [node.id for node in neighbors] == ["concept:edl"]


# Verifies that missing edge reference fails validation.
def test_missing_edge_reference_fails_validation() -> None:
    cal = GraphNode(id="concept:cal", type=NodeType.CONCEPT, title="CAL")
    broken_edge = GraphEdge(
        source_id="concept:missing",
        target_id=cal.id,
        type=EdgeType.REFERENCES,
    )

    with pytest.raises(ValueError, match="missing node references"):
        KnowledgeGraph.from_parts(nodes=[cal], edges=[broken_edge])
