from cal.context.budget import estimate_tokens, pack_documents_to_budget
from dullahan_shared.schemas.context import ContextDocument, ContextSource

from testing_fakes import WhitespaceTokenCounter

TOKEN_COUNTER = WhitespaceTokenCounter()


# Verifies packing behavior with the external tokenizer explicitly mocked.
def test_pack_documents_to_budget_keeps_ranked_documents_that_fit() -> None:
    documents = [
        ContextDocument(id="doc:1", source=ContextSource.PARENT, text="one two"),
        ContextDocument(id="doc:2", source=ContextSource.PARENT, text="three four"),
        ContextDocument(id="doc:3", source=ContextSource.PARENT, text="five six"),
    ]

    packed = pack_documents_to_budget(
        documents,
        token_budget=4,
        token_counter=TOKEN_COUNTER,
    )

    assert [document.id for document in packed] == ["doc:1", "doc:2"]
    assert estimate_tokens(packed, token_counter=TOKEN_COUNTER) == 4


# Verifies truncation behavior with the external tokenizer explicitly mocked.
def test_pack_documents_to_budget_truncates_first_document_if_needed() -> None:
    documents = [
        ContextDocument(
            id="doc:long",
            source=ContextSource.PARENT,
            text="one two three four five",
        )
    ]

    packed = pack_documents_to_budget(
        documents,
        token_budget=3,
        token_counter=TOKEN_COUNTER,
    )

    assert packed[0].text == "one two three"
    assert packed[0].metadata["truncated"] is True
    assert packed[0].metadata["original_token_count"] == 5
    assert packed[0].metadata["tokenizer_model"] == TOKEN_COUNTER.model_id


# Verifies non-positive budgets without invoking the explicitly mocked tokenizer.
def test_pack_documents_to_budget_returns_empty_for_non_positive_budget() -> None:
    documents = [
        ContextDocument(id="doc:1", source=ContextSource.PARENT, text="one two"),
    ]

    assert (
        pack_documents_to_budget(
            documents,
            token_budget=0,
            token_counter=TOKEN_COUNTER,
        )
        == []
    )
