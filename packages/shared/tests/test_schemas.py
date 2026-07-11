import pytest
from pydantic import ValidationError

from dullahan_shared.ids import new_id
from dullahan_shared.schemas.context import ContextBundle, ContextDocument, ContextSource
from dullahan_shared.schemas.expert import ExpertProfile, ExpertResponse
from dullahan_shared.schemas.graph import EdgeType, GraphEdge, GraphNode, NodeType
from dullahan_shared.schemas.query import QueryEnvelope


def test_query_envelope_accepts_parent_context() -> None:
    query_id = new_id("query")
    context = ContextBundle(
        query_id=query_id,
        documents=[
            ContextDocument(
                id="doc:cal",
                source=ContextSource.WORLD_STATE,
                text="CAL augments subqueries with parent and world-state context.",
                score=0.91,
            )
        ],
    )

    envelope = QueryEnvelope(
        sender_id="agent:root",
        query_id=query_id,
        query="How should CAL construct context?",
        parent_context=context,
    )

    assert envelope.parent_context is not None
    assert "CAL augments" in envelope.parent_context.text


def test_expert_profile_requires_role_context() -> None:
    with pytest.raises(ValidationError):
        ExpertProfile(
            id="expert:empty",
            cluster_id="cluster:cal",
            role_context="",
            model="local-slm",
        )


def test_graph_edge_requires_non_negative_weight() -> None:
    source = GraphNode(id="concept:cal", type=NodeType.CONCEPT, title="CAL")
    target = GraphNode(id="concept:edl", type=NodeType.CONCEPT, title="EDL")

    edge = GraphEdge(
        source_id=source.id,
        target_id=target.id,
        type=EdgeType.CONNECTS_TO,
        weight=0.8,
    )

    assert edge.source_id == "concept:cal"
    assert edge.weight == 0.8


def test_expert_response_omits_context_by_contract() -> None:
    response = ExpertResponse(
        sender_id="agent:root",
        query_id="query:1",
        subquery="What does EDL dispatch?",
        expert_id="expert:edl",
        response="EDL dispatches context-augmented subqueries to expert instances.",
        confidence=0.84,
        cited_context_document_ids=["doc:edl"],
    )

    assert not hasattr(response, "context")
