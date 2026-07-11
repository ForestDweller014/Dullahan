from __future__ import annotations

from cal.api.schemas import (
    AugmentContextRequest,
    AugmentContextResponse,
    BatchAugmentContextRequest,
    BatchAugmentContextResponse,
)
from cal.config import CalConfig
from cal.context.budget import estimate_tokens, pack_documents_to_budget
from cal.context.merge import merge_ranked_documents
from dullahan_shared.retrieval.lexical import LexicalRetriever
from dullahan_shared.schemas.context import ContextBundle
from world_state import LocalWorldStateDB, PostgresWorldStateDB


class ContextAugmentationService:
    def __init__(
        self,
        *,
        world_state: LocalWorldStateDB,
        retriever: LexicalRetriever,
        parent_top_k: int,
        world_top_k: int,
        token_budget: int,
    ) -> None:
        self.world_state = world_state
        self.retriever = retriever
        self.parent_top_k = parent_top_k
        self.world_top_k = world_top_k
        self.token_budget = token_budget

    @classmethod
    def from_config(cls, config: CalConfig) -> ContextAugmentationService:
        if config.world_state_backend == "postgres":
            if not config.postgres_dsn:
                raise ValueError(
                    "WORLD_STATE_POSTGRES_DSN is required when WORLD_STATE_BACKEND=postgres"
                )
            world_state = PostgresWorldStateDB.from_graph_memory(
                dsn=config.postgres_dsn,
                repo_root=config.repo_root,
                graph_dir=config.resolved_graph_dir,
                table_name=config.postgres_table_name,
            )
        else:
            world_state = LocalWorldStateDB.from_graph_memory(
                repo_root=config.repo_root,
                graph_dir=config.resolved_graph_dir,
            )
        return cls(
            world_state=world_state,
            retriever=LexicalRetriever(),
            parent_top_k=config.parent_top_k,
            world_top_k=config.world_top_k,
            token_budget=config.token_budget,
        )

    def augment(self, request: AugmentContextRequest) -> AugmentContextResponse:
        parent_ranked = self.retriever.rank(
            request.subquery,
            request.parent_context.documents,
            text_of=lambda document: document.text,
            id_of=lambda document: document.id,
            top_k=self.parent_top_k,
        )
        world_ranked = self.retriever.rank(
            request.subquery,
            self.world_state.search(request.subquery, top_k=self.world_top_k),
            text_of=lambda document: document.text,
            id_of=lambda document: document.id,
            top_k=self.world_top_k,
        )

        ranked_documents = [(item.item, item.score) for item in parent_ranked] + [
            (item.item, item.score) for item in world_ranked
        ]
        merged_documents = merge_ranked_documents(ranked_documents)
        documents = pack_documents_to_budget(merged_documents, token_budget=self.token_budget)
        candidate_token_count = estimate_tokens(merged_documents)
        selected_token_count = estimate_tokens(documents)
        reduction_percent = (
            round((1 - (selected_token_count / candidate_token_count)) * 100, 2)
            if candidate_token_count > 0
            else 0.0
        )

        return AugmentContextResponse(
            subquery=request.subquery,
            context=ContextBundle(
                query_id=request.query_id or request.parent_context.query_id,
                documents=documents,
                token_budget=self.token_budget,
                metadata={
                    "parent_candidate_count": len(parent_ranked),
                    "world_candidate_count": len(world_ranked),
                    "candidate_token_count": candidate_token_count,
                    "selected_token_count": selected_token_count,
                    "context_reduction_percent": reduction_percent,
                },
            ),
        )

    def augment_batch(self, request: BatchAugmentContextRequest) -> BatchAugmentContextResponse:
        return BatchAugmentContextResponse(
            responses=[self.augment(item) for item in request.requests]
        )
