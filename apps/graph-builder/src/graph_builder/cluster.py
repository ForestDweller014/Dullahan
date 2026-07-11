from __future__ import annotations

from pathlib import Path

from dullahan_kg.algorithms import partition_by_k
from dullahan_kg.storage.yaml_graph_store import YamlGraphStore


def generate_clusters(*, graph_dir: Path, k: int, cluster_prefix: str = "cluster:auto") -> None:
    store = YamlGraphStore(graph_dir)
    graph = store.load()
    graph.clusters = {
        cluster.id: cluster
        for cluster in partition_by_k(graph, k=k, cluster_prefix=cluster_prefix)
    }
    store.save(graph)
