from __future__ import annotations

from pathlib import Path

from dullahan_shared.embeddings import EmbeddingModel, cosine_similarity
from dullahan_shared.schemas.context import ContextDocument
from pydantic import BaseModel, Field


class VectorIndexEntry(BaseModel):
    document: ContextDocument
    embedding: list[float]


class VectorSearchResult(BaseModel):
    document: ContextDocument
    score: float = Field(ge=-1.0, le=1.0)


class LocalVectorIndex(BaseModel):
    dimensions: int
    embedding_model_id: str = "legacy-hashing"
    entries: list[VectorIndexEntry] = Field(default_factory=list)

    @classmethod
    def build(
        cls,
        documents: list[ContextDocument],
        embedding_model: EmbeddingModel,
    ) -> LocalVectorIndex:
        embeddings = embedding_model.embed_many([document.text for document in documents])
        return cls(
            dimensions=embedding_model.dimensions,
            embedding_model_id=embedding_model.model_id,
            entries=[
                VectorIndexEntry(
                    document=document,
                    embedding=embedding,
                )
                for document, embedding in zip(documents, embeddings, strict=True)
            ],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LocalVectorIndex:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def search(
        self,
        query: str,
        *,
        embedding_model: EmbeddingModel,
        top_k: int,
    ) -> list[VectorSearchResult]:
        if top_k <= 0:
            return []

        query_embedding = embedding_model.embed(query)
        ranked = [
            VectorSearchResult(
                document=entry.document,
                score=cosine_similarity(query_embedding, entry.embedding),
            )
            for entry in self.entries
        ]
        return [
            result
            for result in sorted(
                ranked,
                key=lambda item: (-item.score, item.document.id),
            )
            if result.score > 0
        ][:top_k]
