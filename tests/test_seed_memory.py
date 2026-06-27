from pathlib import Path

import yaml

from dullahan_kg.storage.yaml_graph_store import YamlGraphStore


ROOT = Path(__file__).resolve().parents[1]


def test_seed_memory_graph_loads() -> None:
    graph = YamlGraphStore(ROOT / "memory" / "graph").load()

    assert graph.get_node("concept:cal").cluster_id == "cluster:context_memory"
    assert graph.get_node("concept:edl").cluster_id == "cluster:expert_dispatch"
    assert len(graph.edges) >= 6


def test_seed_memory_documents_exist() -> None:
    graph = YamlGraphStore(ROOT / "memory" / "graph").load()
    paths = []

    for node in graph.nodes.values():
        paths.extend(node.document_paths)
    for cluster in graph.clusters.values():
        paths.extend(cluster.document_paths)

    missing = [path for path in paths if not (ROOT / path).exists()]

    assert missing == []


def test_seed_experts_reference_existing_clusters_and_documents() -> None:
    graph = YamlGraphStore(ROOT / "memory" / "graph").load()
    with (ROOT / "memory" / "graph" / "experts.yaml").open("r", encoding="utf-8") as stream:
        expert_data = yaml.safe_load(stream)

    for expert in expert_data["experts"]:
        assert expert["cluster_id"] in graph.clusters
        assert (ROOT / expert["role_context_path"]).exists()
