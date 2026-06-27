from __future__ import annotations

from dullahan_shared.ids import new_id
from dullahan_shared.schemas.query import QueryEnvelope
from agent_runtime.planning.provider import (
    DeterministicPlannerProvider,
    PlannerProvider,
    PlannerRequest,
)


class DeterministicSubqueryGenerator:
    def __init__(self, provider: PlannerProvider | None = None) -> None:
        self.provider = provider or DeterministicPlannerProvider()

    def generate(
        self,
        parent_query: QueryEnvelope,
        *,
        max_breadth: int,
    ) -> list[QueryEnvelope]:
        plan = self.provider.plan(
            PlannerRequest(
                parent_query=parent_query,
                max_breadth=max_breadth,
            )
        )

        return [
            QueryEnvelope(
                sender_id=parent_query.query_id,
                query_id=new_id("query"),
                query=f"{candidate} Parent query: {parent_query.query}",
                parent_context=parent_query.parent_context,
                depth=parent_query.depth + 1,
                metadata={
                    "generated_by": plan.provider,
                    "parent_query_id": parent_query.query_id,
                },
            )
            for candidate in plan.subqueries[:max_breadth]
        ]
