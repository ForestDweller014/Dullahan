from pathlib import Path

from cal.api.schemas import AugmentContextRequest, BatchAugmentContextRequest
from cal.config import CalConfig
from cal.context.budget import estimate_tokens
from cal.service import ContextAugmentationService
from dullahan_shared.schemas.context import ContextBundle, ContextDocument, ContextSource


ROOT = Path(__file__).resolve().parents[3]


def build_service() -> ContextAugmentationService:
    return ContextAugmentationService.from_config(
        CalConfig(
            repo_root=ROOT,
            parent_top_k=2,
            world_top_k=3,
            token_budget=2048,
        )
    )


def test_cal_merges_parent_and_graph_context() -> None:
    parent_context = ContextBundle(
        query_id="query:root",
        documents=[
            ContextDocument(
                id="parent:runtime",
                source=ContextSource.PARENT,
                text="The agent runtime asks CAL for context before calling EDL.",
            ),
            ContextDocument(
                id="parent:unrelated",
                source=ContextSource.PARENT,
                text="This paragraph discusses unrelated deployment notes.",
            ),
        ],
    )
    request = AugmentContextRequest(
        sender_id="agent:root",
        query_id="query:child",
        subquery="How does CAL retrieve context for agent runtime subqueries?",
        parent_context=parent_context,
    )

    response = build_service().augment(request)
    document_ids = {document.id for document in response.context.documents}

    assert response.subquery == request.subquery
    assert "parent:runtime" in document_ids
    assert "world-node-doc:concept:cal" in document_ids
    assert response.context.query_id == "query:child"
    assert response.context.token_budget == 2048
    assert response.context.metadata["candidate_token_count"] >= response.context.metadata[
        "selected_token_count"
    ]
    assert "context_reduction_percent" in response.context.metadata
    assert estimate_tokens(response.context.documents) <= response.context.token_budget


def test_cal_enforces_context_token_budget() -> None:
    parent_context = ContextBundle(
        query_id="query:root",
        documents=[
            ContextDocument(
                id="parent:long",
                source=ContextSource.PARENT,
                text=" ".join(["context"] * 20),
            )
        ],
    )
    service = ContextAugmentationService.from_config(
        CalConfig(
            repo_root=ROOT,
            parent_top_k=1,
            world_top_k=0,
            token_budget=5,
        )
    )

    response = service.augment(
        AugmentContextRequest(
            sender_id="agent:root",
            subquery="context",
            parent_context=parent_context,
        )
    )

    assert estimate_tokens(response.context.documents) <= 5
    assert response.context.documents[0].metadata["truncated"] is True


def test_cal_returns_empty_context_when_no_documents_match() -> None:
    parent_context = ContextBundle(
        query_id="query:root",
        documents=[
            ContextDocument(
                id="parent:math",
                source=ContextSource.PARENT,
                text="Prime factorization over integers.",
            )
        ],
    )
    request = AugmentContextRequest(
        sender_id="agent:root",
        subquery="zzzz qqqq xxxx",
        parent_context=parent_context,
    )

    response = build_service().augment(request)

    assert response.context.documents == []


def test_cal_batch_augmentation_preserves_request_order() -> None:
    parent_context = ContextBundle(
        query_id="query:root",
        documents=[
            ContextDocument(
                id="parent:cal",
                source=ContextSource.PARENT,
                text="CAL retrieves and packs context.",
            ),
            ContextDocument(
                id="parent:edl",
                source=ContextSource.PARENT,
                text="EDL routes subqueries to experts.",
            ),
        ],
    )

    response = build_service().augment_batch(
        BatchAugmentContextRequest(
            requests=[
                AugmentContextRequest(
                    sender_id="query:root",
                    subquery="How does CAL pack context?",
                    parent_context=parent_context,
                ),
                AugmentContextRequest(
                    sender_id="query:root",
                    subquery="How does EDL route experts?",
                    parent_context=parent_context,
                ),
            ]
        )
    )

    assert [item.subquery for item in response.responses] == [
        "How does CAL pack context?",
        "How does EDL route experts?",
    ]
    assert len(response.responses) == 2
