from __future__ import annotations

from dullahan_shared.schemas.expert import ExpertProfile, ExpertResponse

from edl.api.schemas import DispatchRequest
from edl.dispatch.attention_router import ExpertRoute
from edl.execution.model_provider import ModelProvider, ModelRequest
from edl.execution.prompt import ExpertPromptBuilder


class ExpertRunner:
    def __init__(
        self,
        *,
        prompt_builder: ExpertPromptBuilder,
        model_provider: ModelProvider,
        max_tokens: int = 512,
        model_override: str | None = None,
    ) -> None:
        self.prompt_builder = prompt_builder
        self.model_provider = model_provider
        self.max_tokens = max_tokens
        self.model_override = model_override

    def run(
        self,
        request: DispatchRequest,
        expert: ExpertProfile,
        route: ExpertRoute,
    ) -> ExpertResponse:
        cited_document_ids = [document.id for document in request.context.documents[:5]]
        prompt = self.prompt_builder.build(request, expert)
        selected_model = self.model_override or expert.model
        model_result = self.model_provider.complete(
            ModelRequest(
                model=selected_model,
                prompt=prompt,
                max_tokens=self.max_tokens,
            )
        )

        return ExpertResponse(
            sender_id=request.sender_id,
            query_id=request.query_id,
            subquery=request.subquery,
            expert_id=expert.id,
            response=model_result.text,
            confidence=route.probability,
            cited_context_document_ids=cited_document_ids,
            routing_metadata={
                "route_raw_score": route.score,
                "route_probability": route.probability,
                "candidate_count": len(route.distribution),
                "attention_scoring": "embedding_cosine",
                "model": selected_model,
                "expert_model": expert.model,
                "model_provider": model_result.provider,
                "model_token_count": model_result.token_count,
            },
        )
