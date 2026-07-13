from pathlib import Path

from dullahan_shared.schemas.context import ContextSource
from world_state import LocalWorldStateDB

from testing_fakes import KeywordEmbeddingModel

ROOT = Path(__file__).resolve().parents[3]


# Verifies local vector ranking with the external semantic embedder explicitly mocked.
def test_world_state_search_returns_ranked_memory_documents(tmp_path: Path) -> None:
    db = LocalWorldStateDB.from_graph_memory(
        repo_root=ROOT,
        graph_dir=ROOT / "memory" / "graph",
        index_path=tmp_path / "world-state.json",
        embedding_model=KeywordEmbeddingModel(),
    )

    results = db.search("CAL retrieves world-state context", top_k=3)

    assert results
    assert results[0].source == ContextSource.WORLD_STATE
    assert results[0].score is not None
    assert any("cal" in document.id for document in results)


# Verifies local vector top-k behavior with the external semantic embedder mocked.
def test_world_state_search_respects_top_k(tmp_path: Path) -> None:
    db = LocalWorldStateDB.from_graph_memory(
        repo_root=ROOT,
        graph_dir=ROOT / "memory" / "graph",
        index_path=tmp_path / "world-state.json",
        embedding_model=KeywordEmbeddingModel(),
    )

    results = db.search("expert dispatch routing context graph", top_k=2)

    assert len(results) <= 2


# Verifies index persistence and reuse with the external semantic embedder mocked.
def test_world_state_builds_and_reuses_persistent_vector_index(tmp_path: Path) -> None:
    index_path = tmp_path / "world_state.json"
    db = LocalWorldStateDB.from_graph_memory(
        repo_root=ROOT,
        graph_dir=ROOT / "memory" / "graph",
        index_path=index_path,
        embedding_model=KeywordEmbeddingModel(),
    )

    first_results = db.search("knowledge graph memory nodes edges", top_k=2)
    second_results = db.search("knowledge graph memory nodes edges", top_k=2)

    assert index_path.exists()
    assert [document.id for document in first_results] == [
        document.id for document in second_results
    ]
