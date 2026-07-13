import json
from pathlib import Path

import yaml
from dullahan_kg.graph import GraphCluster, KnowledgeGraph
from dullahan_kg.storage.yaml_graph_store import YamlGraphStore
from dullahan_shared.schemas.graph import EdgeType, GraphEdge, GraphNode, NodeType
from graph_builder.cluster import generate_clusters
from graph_builder.experts import generate_experts_from_clusters
from graph_builder.graphify import GraphifyConfig, import_graphify_json
from world_state import LocalWorldStateDB

from testing_fakes import KeywordEmbeddingModel


# Verifies that cluster generation rewrites clusters.yaml with K-bounded assignments.
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


# Verifies that generate experts from clusters writes registry and role docs.
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


# Verifies that generate experts from clusters preserves existing role context path.
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

    assert (
        experts["experts"][0]["role_context_path"]
        == "memory/documents/clusters/cluster__manual.md"
    )


# Verifies Graphify import while the external semantic embedder is explicitly mocked.
def test_import_graphify_json_builds_clusters_and_experts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    graphify_json = tmp_path / "graphify-out" / "graph.json"
    graphify_json.parent.mkdir()
    graphify_json.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "macro.md",
                        "label": "Macro Rates Note",
                        "source_file": "macro.md",
                        "description": "Rates inflation duration curve steepener.",
                    },
                    {
                        "id": "rates.md",
                        "label": "Rates Note",
                        "source_file": "rates.md",
                        "description": "Treasury curve duration policy.",
                    },
                    {
                        "id": "portfolio",
                        "label": "Portfolio Concept",
                        "type": "concept",
                    },
                ],
                "edges": [
                    {
                        "source": "portfolio",
                        "target": "macro.md",
                        "relation": "contains",
                        "weight": 1.0,
                    },
                    {
                        "source": "macro.md",
                        "target": "rates.md",
                        "relation": "semantic_similarity",
                        "confidence": 0.82,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    graph = import_graphify_json(
        config=GraphifyConfig(
            source_path=tmp_path / "source",
            repo_root=repo_root,
            graph_dir=repo_root / "memory" / "graph",
            documents_dir=repo_root / "memory" / "documents" / "nodes",
            k=2,
        ),
        graphify_json_path=graphify_json,
        embedding_model=KeywordEmbeddingModel(),
    )

    graph_yaml = repo_root / "memory" / "graph" / "graph.yaml"
    clusters_yaml = repo_root / "memory" / "graph" / "clusters.yaml"
    experts_yaml = repo_root / "memory" / "graph" / "experts.yaml"
    index_path = (
        repo_root
        / "memory"
        / "world_state"
        / "indexes"
        / "local.test_keyword_embedding.json"
    )

    assert graph_yaml.exists()
    assert clusters_yaml.exists()
    assert experts_yaml.exists()
    assert index_path.exists()
    assert len(graph.nodes) == 3
    assert graph.clusters
    assert any(edge.type == EdgeType.OWNS for edge in graph.edges)
    assert any(edge.type == EdgeType.SEMANTICALLY_RELATED for edge in graph.edges)

    file_nodes = [node for node in graph.nodes.values() if node.type == NodeType.FILE]
    assert file_nodes
    assert all(node.document_paths for node in file_nodes)
    assert (repo_root / file_nodes[0].document_paths[0]).exists()

    experts = yaml.safe_load(experts_yaml.read_text(encoding="utf-8"))
    assert len(experts["experts"]) == len(graph.clusters)

    world_state = LocalWorldStateDB.from_graph_memory(
        repo_root=repo_root,
        graph_dir=repo_root / "memory" / "graph",
        embedding_model=KeywordEmbeddingModel(),
    )
    assert world_state.search("duration curve", top_k=1)
