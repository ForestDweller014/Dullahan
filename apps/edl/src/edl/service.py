from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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
    DeterministicLocalSlmProvider,
    ModelProvider,
    OpenAICompatibleHttpProvider,
)
from edl.execution.prompt import ExpertPromptBuilder


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

    @classmethod
    def from_config(cls, config: EdlConfig) -> ExpertDispatchService:
        return cls(
            registry=ExpertRegistry(
                repo_root=config.repo_root,
                experts_path=config.resolved_experts_path,
            ),
            router=AttentionRouter(
                min_score_threshold=config.min_score_threshold,
            ),
            runner=ExpertRunner(
                prompt_builder=ExpertPromptBuilder(),
                model_provider=cls._build_model_provider(config),
            ),
            max_dispatch_concurrency=config.max_dispatch_concurrency,
        )

    @staticmethod
    def _build_model_provider(config: EdlConfig) -> ModelProvider:
        if config.model_provider == "deterministic":
            return DeterministicLocalSlmProvider()
        if config.model_provider == "http":
            return OpenAICompatibleHttpProvider(
                base_url=config.model_base_url,
                timeout_seconds=config.model_timeout_seconds,
            )
        raise ValueError(f"unknown EDL model provider: {config.model_provider}")

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
        return self.runner.run(
            request=request,
            expert=route.expert,
            route=route,
        )
