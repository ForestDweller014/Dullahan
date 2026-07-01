from __future__ import annotations

import pytest

from dullahan_shared.embeddings import HashingEmbeddingModel
from dullahan_shared.schemas.context import ContextDocument, ContextSource
from world_state.postgres import (
    PostgresWorldStateConfig,
    PostgresWorldStateDB,
    _to_pgvector,
)


class FakeDocumentSource:
    def load_documents(self) -> list[ContextDocument]:
        return [
            ContextDocument(
                id="world-node-doc:alpha",
                source=ContextSource.GRAPH_NODE,
                text="Alpha risk context for portfolio hedging.",
                metadata={"node_id": "concept:alpha"},
            )
        ]


class FakeCursor:
    def __init__(self, rows: list[tuple] | None = None) -> None:
        self.rows = rows or []
        self.executed: list[tuple[str, tuple | None]] = []
        self.executemany_calls: list[tuple[str, list[tuple]]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executed.append((sql, params))

    def executemany(self, sql: str, params: list[tuple]) -> None:
        self.executemany_calls.append((sql, params))

    def fetchall(self) -> list[tuple]:
        return self.rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_instance = cursor
        self.commit_count = 0

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commit_count += 1


def test_postgres_world_state_search_uses_pgvector_similarity() -> None:
    schema_cursor = FakeCursor()
    search_cursor = FakeCursor(
        rows=[
            (
                "world-node-doc:alpha",
                "graph_node",
                "Alpha risk context for portfolio hedging.",
                {"node_id": "concept:alpha"},
                0.91,
            )
        ]
    )
    connections = [FakeConnection(schema_cursor), FakeConnection(search_cursor)]

    db = PostgresWorldStateDB(
        config=PostgresWorldStateConfig(dsn="postgresql://example/db"),
        document_source=FakeDocumentSource(),
        embedding_model=HashingEmbeddingModel(),
        connect=lambda _dsn: connections.pop(0),
    )

    results = db.search("portfolio risk hedge", top_k=3)

    assert results[0].id == "world-node-doc:alpha"
    assert results[0].source == ContextSource.WORLD_STATE
    assert results[0].score == 0.91
    assert "embedding <=>" in search_cursor.executed[0][0]


def test_postgres_world_state_rebuild_writes_graph_documents_to_pgvector() -> None:
    schema_cursor = FakeCursor()
    rebuild_cursor = FakeCursor()
    connections = [FakeConnection(schema_cursor), FakeConnection(rebuild_cursor)]

    db = PostgresWorldStateDB(
        config=PostgresWorldStateConfig(dsn="postgresql://example/db"),
        document_source=FakeDocumentSource(),
        embedding_model=HashingEmbeddingModel(),
        connect=lambda _dsn: connections.pop(0),
    )

    inserted = db.rebuild_index()

    assert inserted == 1
    assert rebuild_cursor.executemany_calls
    sql, params = rebuild_cursor.executemany_calls[0]
    assert "INSERT INTO world_state_documents" in sql
    assert params[0][0] == "world-node-doc:alpha"
    assert params[0][4].startswith("[")


def test_postgres_world_state_rejects_unsafe_table_names() -> None:
    with pytest.raises(ValueError, match="table name"):
        PostgresWorldStateDB(
            config=PostgresWorldStateConfig(
                dsn="postgresql://example/db",
                table_name="world_state_documents; drop table users",
            ),
            document_source=FakeDocumentSource(),
            embedding_model=HashingEmbeddingModel(),
        )


def test_pgvector_literal_formats_embedding_for_sql_cast() -> None:
    assert _to_pgvector([0.5, -0.25]) == "[0.50000000,-0.25000000]"
