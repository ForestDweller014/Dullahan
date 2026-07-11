from __future__ import annotations

from pathlib import Path

from dullahan_kg.storage.yaml_graph_store import YamlGraphStore
from dullahan_shared.schemas.context import ContextDocument, ContextSource


class GraphDocumentSource:
    def __init__(self, repo_root: Path, graph_dir: Path) -> None:
        self.repo_root = repo_root
        self.graph_dir = graph_dir

    def load_documents(self) -> list[ContextDocument]:
        graph = YamlGraphStore(self.graph_dir).load()
        documents: list[ContextDocument] = []

        for node in graph.nodes.values():
            for document_path in node.document_paths:
                documents.append(
                    self._read_document(
                        document_id=f"world-node-doc:{node.id}",
                        path=document_path,
                        source=ContextSource.GRAPH_NODE,
                        metadata={"node_id": node.id, "title": node.title},
                    )
                )

        for cluster in graph.clusters.values():
            for document_path in cluster.document_paths:
                documents.append(
                    self._read_document(
                        document_id=f"world-cluster-doc:{cluster.id}",
                        path=document_path,
                        source=ContextSource.GRAPH_CLUSTER,
                        metadata={"cluster_id": cluster.id, "title": cluster.title},
                    )
                )

        return documents

    def _read_document(
        self,
        *,
        document_id: str,
        path: str,
        source: ContextSource,
        metadata: dict[str, str | int | float | bool],
    ) -> ContextDocument:
        resolved_path = self.repo_root / path
        text = resolved_path.read_text(encoding="utf-8")
        return ContextDocument(
            id=document_id,
            source=source,
            text=text,
            metadata={**metadata, "path": path},
        )
