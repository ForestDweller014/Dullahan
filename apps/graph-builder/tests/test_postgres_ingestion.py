from __future__ import annotations

import sys
import types

from graph_builder.graphify import export_postgres_context


class FakeCursor:
    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str) -> None:
        self.query = query

    def fetchall(self) -> list[dict]:
        return [
            {
                "id": "macro-note-1",
                "title": "Macro Note",
                "content": "Rates volatility moved higher before CPI.",
                "metadata": {"domain": "trading"},
            }
        ]


class FakeConnection:
    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return FakeCursor()


class FakePsycopg:
    @staticmethod
    def connect(*args: object, **kwargs: object) -> FakeConnection:
        return FakeConnection()


def test_export_postgres_context_writes_markdown_collection(tmp_path, monkeypatch) -> None:
    rows_module = types.SimpleNamespace(dict_row=object())
    monkeypatch.setitem(sys.modules, "psycopg", FakePsycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", rows_module)

    export_dir = export_postgres_context(
        dsn="postgresql://example/db",
        query="select id, title, content, metadata from notes",
        export_dir=tmp_path / "postgres_context",
    )

    exported = export_dir / "macro_note_1.md"
    assert exported.exists()
    text = exported.read_text(encoding="utf-8")
    assert "# Macro Note" in text
    assert "Rates volatility moved higher before CPI." in text
    assert '"domain": "trading"' in text
