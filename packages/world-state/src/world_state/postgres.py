from __future__ import annotations

import json
import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from dullahan_shared.embeddings import HashingEmbeddingModel
from dullahan_shared.schemas.context import ContextDocument, ContextSource
from world_state.graph_documents import GraphDocumentSource


TABLE_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?$")


class PostgresWorldStateConfig(BaseModel):
    dsn: str = Field(min_length=1)
    table_name: str = "world_state_documents"
    dimensions: int = Field(default=128, gt=0)


class PostgresWorldStateDB:
    """PostgreSQL + pgvector-backed long-term memory store for CAL retrieval."""

    def __init__(
        self,
        *,
        config: PostgresWorldStateConfig,
        document_source: GraphDocumentSource,
        embedding_model: HashingEmbeddingModel,
        connect: Callable[..., object] | None = None,
    ) -> None:
        self.config = config
        self.document_source = document_source
        self.embedding_model = embedding_model
        self._connect = connect
        self._table_name = _validate_table_name(config.table_name)

    @classmethod
    def from_graph_memory(
        cls,
        *,
        dsn: str,
        repo_root,
        graph_dir,
        table_name: str = "world_state_documents",
        dimensions: int = 128,
    ) -> PostgresWorldStateDB:
        embedding_model = HashingEmbeddingModel(dimensions=dimensions)
        return cls(
            config=PostgresWorldStateConfig(
                dsn=dsn,
                table_name=table_name,
                dimensions=embedding_model.dimensions,
            ),
            document_source=GraphDocumentSource(repo_root=repo_root, graph_dir=graph_dir),
            embedding_model=embedding_model,
        )

    def search(self, query: str, *, top_k: int) -> list[ContextDocument]:
        if top_k <= 0:
            return []

        self.ensure_schema()
        query_vector = _to_pgvector(self.embedding_model.embed(query))
        with self._connect_to_postgres() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    (
                        "SELECT id, source, text, metadata, "
                        "GREATEST(0, 1 - (embedding <=> %s::vector)) AS score "
                        f"FROM {self._table_name} "
                        "ORDER BY embedding <=> %s::vector "
                        "LIMIT %s"
                    ),
                    (query_vector, query_vector, top_k),
                )
                rows = cursor.fetchall()

        return [
            ContextDocument(
                id=str(row[0]),
                source=ContextSource.WORLD_STATE,
                text=str(row[2]),
                score=round(float(row[4]), 6),
                metadata={
                    **_metadata_dict(row[3]),
                    "postgres_source": str(row[1]),
                    "postgres_table": self._table_name,
                },
            )
            for row in rows
            if float(row[4]) > 0
        ]

    def rebuild_index(self) -> int:
        self.ensure_schema()
        documents = self.document_source.load_documents()
        with self._connect_to_postgres() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {self._table_name}")
                cursor.executemany(
                    (
                        f"INSERT INTO {self._table_name} "
                        "(id, source, text, metadata, embedding) "
                        "VALUES (%s, %s, %s, %s::jsonb, %s::vector)"
                    ),
                    [
                        (
                            document.id,
                            document.source.value,
                            document.text,
                            json.dumps(document.metadata, sort_keys=True),
                            _to_pgvector(self.embedding_model.embed(document.text)),
                        )
                        for document in documents
                    ],
                )
            connection.commit()
        return len(documents)

    def ensure_schema(self) -> None:
        with self._connect_to_postgres() as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    (
                        f"CREATE TABLE IF NOT EXISTS {self._table_name} ("
                        "id text PRIMARY KEY, "
                        "source text NOT NULL, "
                        "text text NOT NULL, "
                        "metadata jsonb NOT NULL DEFAULT '{}'::jsonb, "
                        f"embedding vector({self.embedding_model.dimensions}) NOT NULL"
                        ")"
                    )
                )
                cursor.execute(
                    (
                        f"CREATE INDEX IF NOT EXISTS {_index_name(self._table_name)} "
                        f"ON {self._table_name} USING ivfflat "
                        "(embedding vector_cosine_ops)"
                    )
                )
            connection.commit()

    def _connect_to_postgres(self):
        if self._connect is not None:
            return self._connect(self.config.dsn)
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL WorldStateDB requires psycopg. "
                "Install project dependencies with `python -m pip install -e \".[dev]\"`."
            ) from exc
        return psycopg.connect(self.config.dsn)


def _validate_table_name(table_name: str) -> str:
    if not TABLE_NAME_PATTERN.match(table_name):
        raise ValueError(
            "PostgreSQL table name must be an identifier or schema-qualified identifier"
        )
    return table_name


def _index_name(table_name: str) -> str:
    return f"{table_name.replace('.', '_')}_embedding_idx"


def _to_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"


def _metadata_dict(value: object) -> dict[str, str | int | float | bool]:
    if isinstance(value, dict):
        return {
            str(key): item
            for key, item in value.items()
            if isinstance(item, str | int | float | bool)
        }
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return _metadata_dict(parsed)
    return {}
