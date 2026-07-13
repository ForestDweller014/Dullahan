from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Condition

from dullahan_shared.embeddings import EmbeddingModel, OpenAICompatibleEmbeddingModel
from dullahan_shared.schemas.expert import ExpertProfile, ExpertResponse

from edl.api.schemas import (
    BatchDispatchRequest,
    BatchDispatchResponse,
    DispatchRequest,
    DispatchResponse,
)
from edl.config import EdlConfig
from edl.dispatch.attention_router import AttentionRouter
from edl.dispatch.expert_registry import ExpertRegistry
from edl.execution.expert_runner import ExpertRunner
from edl.execution.model_provider import (
    ModelProvider,
    OpenAICompatibleHttpProvider,
)
from edl.execution.prompt import ExpertPromptBuilder


class _ExpertConcurrencyLimiter:
    def __init__(self) -> None:
        self._condition = Condition()
        self._active_instances: dict[str, int] = {}

    @contextmanager
    def reserve(self, expert: ExpertProfile) -> Iterator[None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._active_instances.get(expert.id, 0) < expert.max_concurrency
            )
            self._active_instances[expert.id] = self._active_instances.get(expert.id, 0) + 1

        try:
            yield
        finally:
            with self._condition:
                remaining = self._active_instances[expert.id] - 1
                if remaining:
                    self._active_instances[expert.id] = remaining
                else:
                    del self._active_instances[expert.id]
                self._condition.notify_all()


class ExpertDispatchService:
    def __init__(
        self,
        *,
        registry: ExpertRegistry,
        router: AttentionRouter,
        runner: ExpertRunner,
        max_dispatch_concurrency: int,
    ) -> None:
        self.registry = registry
        self.router = router
        self.runner = runner
        self.max_dispatch_concurrency = max_dispatch_concurrency
        self._expert_concurrency = _ExpertConcurrencyLimiter()

    @classmethod
    def from_config(
        cls,
        config: EdlConfig,
        *,
        model_provider: ModelProvider | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> ExpertDispatchService:
        return cls(
            registry=ExpertRegistry(
                repo_root=config.repo_root,
                experts_path=config.resolved_experts_path,
            ),
            router=AttentionRouter(
                embedding_model=embedding_model
                or OpenAICompatibleEmbeddingModel(
                    base_url=config.model_base_url,
                    model=config.embedding_model,
                    dimensions=config.embedding_dimensions,
                    timeout_seconds=config.model_timeout_seconds,
                ),
                min_score_threshold=config.min_score_threshold,
            ),
            runner=ExpertRunner(
                prompt_builder=ExpertPromptBuilder(),
                model_provider=model_provider or cls._build_model_provider(config),
                max_tokens=config.model_max_tokens,
            ),
            max_dispatch_concurrency=config.max_dispatch_concurrency,
        )

    @staticmethod
    def _build_model_provider(config: EdlConfig) -> ModelProvider:
        return OpenAICompatibleHttpProvider(
            base_url=config.model_base_url,
            timeout_seconds=config.model_timeout_seconds,
        )

    def dispatch(self, request: DispatchRequest) -> DispatchResponse:
        experts = self.registry.load()
        return DispatchResponse(response=self._dispatch_with_experts(request, experts))

    def dispatch_batch(self, request: BatchDispatchRequest) -> BatchDispatchResponse:
        experts = self.registry.load()
        worker_count = min(self.max_dispatch_concurrency, len(request.requests))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self._dispatch_with_experts, item, experts)
                for item in request.requests
            ]
            return BatchDispatchResponse(responses=[future.result() for future in futures])

    def _dispatch_with_experts(
        self,
        request: DispatchRequest,
        experts: list[ExpertProfile],
    ) -> ExpertResponse:
        route = self.router.select(request.subquery, experts)
        with self._expert_concurrency.reserve(route.expert):
            return self.runner.run(
                request=request,
                expert=route.expert,
                route=route,
            )
