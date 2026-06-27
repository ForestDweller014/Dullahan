from __future__ import annotations

from dullahan_shared.schemas.context import ContextDocument


def merge_ranked_documents(
    documents: list[tuple[ContextDocument, float]],
) -> list[ContextDocument]:
    merged: dict[str, ContextDocument] = {}

    for document, score in sorted(documents, key=lambda item: (-item[1], item[0].id)):
        if document.id in merged:
            continue
        merged[document.id] = document.model_copy(update={"score": round(score, 6)})

    return list(merged.values())
