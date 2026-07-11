from cal.context.budget import estimate_tokens, pack_documents_to_budget
from dullahan_shared.schemas.context import ContextDocument, ContextSource


def test_pack_documents_to_budget_keeps_ranked_documents_that_fit() -> None:
    documents = [
        ContextDocument(id="doc:1", source=ContextSource.PARENT, text="one two"),
        ContextDocument(id="doc:2", source=ContextSource.PARENT, text="three four"),
        ContextDocument(id="doc:3", source=ContextSource.PARENT, text="five six"),
    ]

    packed = pack_documents_to_budget(documents, token_budget=4)

    assert [document.id for document in packed] == ["doc:1", "doc:2"]
    assert estimate_tokens(packed) == 4


def test_pack_documents_to_budget_truncates_first_document_if_needed() -> None:
    documents = [
        ContextDocument(
            id="doc:long",
            source=ContextSource.PARENT,
            text="one two three four five",
        )
    ]

    packed = pack_documents_to_budget(documents, token_budget=3)

    assert packed[0].text == "one two three"
    assert packed[0].metadata["truncated"] is True
    assert packed[0].metadata["original_token_estimate"] == 5


def test_pack_documents_to_budget_returns_empty_for_non_positive_budget() -> None:
    documents = [
        ContextDocument(id="doc:1", source=ContextSource.PARENT, text="one two"),
    ]

    assert pack_documents_to_budget(documents, token_budget=0) == []
