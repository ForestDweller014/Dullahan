from dullahan_shared.embeddings import HashingEmbeddingModel, cosine_similarity


def test_hashing_embedding_model_is_deterministic_and_normalized() -> None:
    model = HashingEmbeddingModel(dimensions=16)

    first = model.embed("CAL retrieves context")
    second = model.embed("CAL retrieves context")

    assert first == second
    assert round(cosine_similarity(first, first), 6) == 1.0


def test_hashing_embedding_model_rejects_invalid_dimensions() -> None:
    try:
        HashingEmbeddingModel(dimensions=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected invalid dimensions to raise")
