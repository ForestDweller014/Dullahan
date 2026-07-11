from __future__ import annotations

from math import exp

from pydantic import BaseModel, Field

from dullahan_shared.embeddings import HashingEmbeddingModel, cosine_similarity
from dullahan_shared.schemas.expert import ExpertProfile


class ExpertAttentionScore(BaseModel):
    expert_id: str
    raw_score: float = Field(ge=0.0)
    probability: float = Field(ge=0.0, le=1.0)


class ExpertRoute(BaseModel):
    expert: ExpertProfile
    score: float = Field(ge=0.0)
    probability: float = Field(ge=0.0, le=1.0)
    distribution: list[ExpertAttentionScore]


class AttentionRouter:
    def __init__(
        self,
        embedding_model: HashingEmbeddingModel | None = None,
        min_score_threshold: float = 0.0,
    ) -> None:
        self.embedding_model = embedding_model or HashingEmbeddingModel()
        self.min_score_threshold = min_score_threshold

    def select(self, subquery: str, experts: list[ExpertProfile]) -> ExpertRoute:
        if not experts:
            raise ValueError("cannot route without registered experts")

        subquery_embedding = self.embedding_model.embed(subquery)
        raw_scores = {
            expert.id: max(
                0.0,
                cosine_similarity(
                    subquery_embedding,
                    self.embedding_model.embed(expert.role_context),
                ),
            )
            for expert in experts
        }

        distribution = self._softmax(raw_scores)
        selected_score = max(distribution, key=lambda score: (score.probability, score.raw_score, score.expert_id))
        selected_expert = next(expert for expert in experts if expert.id == selected_score.expert_id)

        if selected_score.raw_score < self.min_score_threshold:
            selected_expert = sorted(experts, key=lambda expert: expert.id)[0]
            selected_score = next(score for score in distribution if score.expert_id == selected_expert.id)

        return ExpertRoute(
            expert=selected_expert,
            score=selected_score.raw_score,
            probability=selected_score.probability,
            distribution=distribution,
        )

    def _softmax(self, raw_scores: dict[str, float]) -> list[ExpertAttentionScore]:
        max_score = max(raw_scores.values(), default=0.0)
        exponentials = {
            expert_id: exp(score - max_score)
            for expert_id, score in raw_scores.items()
        }
        total = sum(exponentials.values()) or 1.0
        return [
            ExpertAttentionScore(
                expert_id=expert_id,
                raw_score=round(raw_scores[expert_id], 6),
                probability=round(exponentials[expert_id] / total, 6),
            )
            for expert_id in sorted(raw_scores)
        ]
