from __future__ import annotations

from dullahan_shared.schemas.expert import ExpertResponse
from dullahan_shared.schemas.query import QueryEnvelope


class ResponseAggregator:
    def aggregate(self, root_query: QueryEnvelope, responses: list[ExpertResponse]) -> str:
        if not responses:
            return f"No expert responses were produced for: {root_query.query}"

        lines = [f"Root query: {root_query.query}", "Expert responses:"]
        for response in responses:
            lines.append(f"- {response.expert_id}: {response.response}")
        return "\n".join(lines)
