from __future__ import annotations

from pathlib import Path

from dullahan_shared.embeddings import EmbeddingModel, OpenAICompatibleEmbeddingModel
from dullahan_shared.schemas.context import ContextDocument, ContextSource

from world_state.graph_documents import GraphDocumentSource
from world_state.vector_index import LocalVectorIndex


class LocalWorldStateDB:
    """Local long-term memory store used by CAL during context augmentation."""

    def __init__(
        self,
        *,
        document_source: GraphDocumentSource,
        embedding_model: EmbeddingModel,
        index_path: Path,
    ) -> None:
        self.document_source = document_source
        self.embedding_model = embedding_model
        self.index_path = index_path

    @classmethod
    def from_graph_memory(
        cls,
        *,
        repo_root: Path,
        graph_dir: Path,
        index_path: Path | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> LocalWorldStateDB:
        selected_embedding_model = embedding_model or _default_embedding_model()
        return cls(
            document_source=GraphDocumentSource(repo_root=repo_root, graph_dir=graph_dir),
            embedding_model=selected_embedding_model,
            index_path=index_path
            or repo_root
            / "memory"
            / "world_state"
            / "indexes"
            / f"local.{_safe_model_id(selected_embedding_model.model_id)}.json",
        )

    def search(self, query: str, *, top_k: int) -> list[ContextDocument]:
        index = self.load_or_build_index()
        ranked = index.search(query, embedding_model=self.embedding_model, top_k=top_k)
        return [
            item.document.model_copy(
                update={
                    "source": self._normalize_source(item.document.source),
                    "score": round(item.score, 6),
                }
            )
            for item in ranked
        ]

    def rebuild_index(self) -> LocalVectorIndex:
        index = LocalVectorIndex.build(
            self.document_source.load_documents(),
            embedding_model=self.embedding_model,
        )
        index.save(self.index_path)
        return index

    def load_or_build_index(self) -> LocalVectorIndex:
        if self.index_path.exists():
            index = LocalVectorIndex.load(self.index_path)
            if (
                index.embedding_model_id == self.embedding_model.model_id
                and index.dimensions == self.embedding_model.dimensions
            ):
                return index
        return self.rebuild_index()

    def _normalize_source(self, source: ContextSource) -> ContextSource:
        if source in {ContextSource.GRAPH_NODE, ContextSource.GRAPH_CLUSTER}:
            return ContextSource.WORLD_STATE
        return source


def _default_embedding_model() -> OpenAICompatibleEmbeddingModel:
    import os

    return OpenAICompatibleEmbeddingModel(
        base_url=os.getenv("DULLAHAN_INFERENCE_BASE_URL", "http://127.0.0.1:30000/v1"),
        model=os.getenv("DULLAHAN_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        dimensions=int(os.getenv("DULLAHAN_EMBEDDING_DIMENSIONS", "1024")),
        timeout_seconds=float(os.getenv("DULLAHAN_INFERENCE_TIMEOUT_SECONDS", "120")),
    )


def _safe_model_id(model_id: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in model_id)
