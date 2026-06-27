from pathlib import Path

import yaml

from dullahan_kg.graph import GraphCluster, KnowledgeGraph
from dullahan_kg.storage.yaml_graph_store import YamlGraphStore
from dullahan_shared.schemas.graph import EdgeType, GraphEdge, GraphNode, NodeType
from graph_builder.cluster import generate_clusters
from graph_builder.experts import generate_experts_from_clusters


def test_generate_clusters_updates_clusters_yaml(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graph"
    store = YamlGraphStore(graph_dir)
    store.save(
        KnowledgeGraph.from_parts(
            nodes=[
                GraphNode(id="concept:a", type=NodeType.CONCEPT, title="A"),
                GraphNode(id="concept:b", type=NodeType.CONCEPT, title="B"),
                GraphNode(id="concept:c", type=NodeType.CONCEPT, title="C"),
            ],
            edges=[
                GraphEdge(
                    source_id="concept:a",
                    target_id="concept:b",
                    type=EdgeType.CONNECTS_TO,
                )
            ],
        )
    )

    generate_clusters(graph_dir=graph_dir, k=2, cluster_prefix="cluster:test")
    graph = store.load()

    assert graph.clusters
    assert all(cluster.id.startswith("cluster:test:") for cluster in graph.clusters.values())
    assert all(len(cluster.node_ids) <= 2 for cluster in graph.clusters.values())


def test_generate_experts_from_clusters_writes_registry_and_role_docs(tmp_path: Path) -> None:
    graph_dir = tmp_path / "memory" / "graph"
    store = YamlGraphStore(graph_dir)
    store.save(
        KnowledgeGraph.from_parts(
            nodes=[
                GraphNode(id="concept:a", type=NodeType.CONCEPT, title="A"),
                GraphNode(id="concept:b", type=NodeType.CONCEPT, title="B"),
            ],
            edges=[],
        )
    )
    generate_clusters(graph_dir=graph_dir, k=2, cluster_prefix="cluster:test")

    generate_experts_from_clusters(graph_dir=graph_dir, repo_root=tmp_path)

    experts = yaml.safe_load((graph_dir / "experts.yaml").read_text(encoding="utf-8"))
    expert = experts["experts"][0]
    role_context_path = tmp_path / expert["role_context_path"]

    assert expert["id"] == "expert:cluster_test_1"
    assert expert["cluster_id"] == "cluster:test:1"
    assert expert["model"] == "local-slm-cluster_test_1"
    assert expert["metadata"]["node_count"] == 2
    assert role_context_path.exists()
    assert "concept:a" in role_context_path.read_text(encoding="utf-8")


def test_generate_experts_from_clusters_preserves_existing_role_context_path(
    tmp_path: Path,
) -> None:
    graph_dir = tmp_path / "memory" / "graph"
    role_doc = tmp_path / "memory" / "documents" / "clusters" / "cluster__manual.md"
    role_doc.parent.mkdir(parents=True)
    role_doc.write_text("# Manual Cluster\n", encoding="utf-8")
    store = YamlGraphStore(graph_dir)
    store.save(
        KnowledgeGraph.from_parts(
            nodes=[GraphNode(id="concept:a", type=NodeType.CONCEPT, title="A")],
            edges=[],
            clusters=[
                GraphCluster(
                    id="cluster:manual",
                    title="Manual Cluster",
                    node_ids=["concept:a"],
                    document_paths=["memory/documents/clusters/cluster__manual.md"],
                )
            ],
        )
    )

    generate_experts_from_clusters(graph_dir=graph_dir, repo_root=tmp_path)

    experts = yaml.safe_load((graph_dir / "experts.yaml").read_text(encoding="utf-8"))

    assert experts["experts"][0]["role_context_path"] == "memory/documents/clusters/cluster__manual.md"
