from __future__ import annotations

from cal.api.schemas import (
    AugmentContextRequest,
    AugmentContextResponse,
    BatchAugmentContextRequest,
    BatchAugmentContextResponse,
)
from cal.config import CalConfig
from cal.context.budget import pack_documents_to_budget
from cal.context.merge import merge_ranked_documents
from dullahan_shared.retrieval.lexical import LexicalRetriever
from dullahan_shared.schemas.context import ContextBundle
from world_state import LocalWorldStateDB


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
        return cls(
            world_state=LocalWorldStateDB.from_graph_memory(
                repo_root=config.repo_root,
                graph_dir=config.resolved_graph_dir,
            ),
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

        documents = pack_documents_to_budget(
            merge_ranked_documents(
                [(item.item, item.score) for item in parent_ranked]
                + [(item.item, item.score) for item in world_ranked]
            ),
            token_budget=self.token_budget,
        )

        return AugmentContextResponse(
            subquery=request.subquery,
            context=ContextBundle(
                query_id=request.query_id or request.parent_context.query_id,
                documents=documents,
                token_budget=self.token_budget,
            ),
        )

    def augment_batch(self, request: BatchAugmentContextRequest) -> BatchAugmentContextResponse:
        return BatchAugmentContextResponse(
            responses=[self.augment(item) for item in request.requests]
        )
