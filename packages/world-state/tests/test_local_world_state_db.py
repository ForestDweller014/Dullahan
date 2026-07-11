from pathlib import Path

from dullahan_shared.schemas.context import ContextSource
from world_state import LocalWorldStateDB


ROOT = Path(__file__).resolve().parents[3]


def test_world_state_search_returns_ranked_memory_documents() -> None:
    db = LocalWorldStateDB.from_graph_memory(
        repo_root=ROOT,
        graph_dir=ROOT / "memory" / "graph",
    )

    results = db.search("CAL retrieves world-state context", top_k=3)

    assert results
    assert results[0].source == ContextSource.WORLD_STATE
    assert results[0].score is not None
    assert any("cal" in document.id for document in results)


def test_world_state_search_respects_top_k() -> None:
    db = LocalWorldStateDB.from_graph_memory(
        repo_root=ROOT,
        graph_dir=ROOT / "memory" / "graph",
    )

    results = db.search("expert dispatch routing context graph", top_k=2)

    assert len(results) <= 2


def test_world_state_builds_and_reuses_persistent_vector_index(tmp_path: Path) -> None:
    index_path = tmp_path / "world_state.json"
    db = LocalWorldStateDB.from_graph_memory(
        repo_root=ROOT,
        graph_dir=ROOT / "memory" / "graph",
        index_path=index_path,
    )

    first_results = db.search("knowledge graph memory nodes edges", top_k=2)
    second_results = db.search("knowledge graph memory nodes edges", top_k=2)

    assert index_path.exists()
    assert [document.id for document in first_results] == [
        document.id for document in second_results
    ]
