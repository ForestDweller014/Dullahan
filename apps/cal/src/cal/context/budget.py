from __future__ import annotations

from dullahan_shared.schemas.context import ContextDocument


def pack_documents_to_budget(
    documents: list[ContextDocument],
    *,
    token_budget: int,
) -> list[ContextDocument]:
    if token_budget <= 0:
        return []

    packed: list[ContextDocument] = []
    used_tokens = 0

    for document in documents:
        tokens = document.text.split()
        if not tokens:
            continue

        remaining = token_budget - used_tokens
        if remaining <= 0:
            break

        if len(tokens) <= remaining:
            packed.append(document)
            used_tokens += len(tokens)
            continue

        if not packed:
            packed.append(
                document.model_copy(
                    update={
                        "text": " ".join(tokens[:remaining]),
                        "metadata": {
                            **document.metadata,
                            "truncated": True,
                            "original_token_estimate": len(tokens),
                        },
                    }
                )
            )
        break

    return packed


def estimate_tokens(documents: list[ContextDocument]) -> int:
    return sum(len(document.text.split()) for document in documents)
