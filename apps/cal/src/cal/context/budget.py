from __future__ import annotations

from dullahan_shared.schemas.context import ContextDocument
from dullahan_shared.tokenization import TokenCounter


def pack_documents_to_budget(
    documents: list[ContextDocument],
    *,
    token_budget: int,
    token_counter: TokenCounter,
) -> list[ContextDocument]:
    if token_budget <= 0:
        return []

    packed: list[ContextDocument] = []
    for document in documents:
        if not document.text:
            continue
        token_count = token_counter.count(document.text)
        combined_text = "\n\n".join(
            [*[item.text for item in packed], document.text]
        )
        if token_counter.count(combined_text) <= token_budget:
            packed.append(document)
            continue
        if not packed:
            truncated_text = _truncate_to_token_budget(
                document.text,
                token_budget=token_budget,
                token_counter=token_counter,
            )
            if truncated_text:
                packed.append(
                    document.model_copy(
                        update={
                            "text": truncated_text,
                            "metadata": {
                                **document.metadata,
                                "truncated": True,
                                "original_token_count": token_count,
                                "tokenizer_model": token_counter.model_id,
                            },
                        }
                    )
                )
        break
    return packed


def estimate_tokens(
    documents: list[ContextDocument],
    *,
    token_counter: TokenCounter,
) -> int:
    if not documents:
        return 0
    return token_counter.count("\n\n".join(document.text for document in documents))


def _truncate_to_token_budget(
    text: str,
    *,
    token_budget: int,
    token_counter: TokenCounter,
) -> str:
    if token_budget <= 0 or not text:
        return ""
    low = 0
    high = len(text)
    while low < high:
        midpoint = (low + high + 1) // 2
        if token_counter.count(text[:midpoint]) <= token_budget:
            low = midpoint
        else:
            high = midpoint - 1
    return text[:low].rstrip()
