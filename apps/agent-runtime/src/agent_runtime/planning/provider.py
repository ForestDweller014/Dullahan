from __future__ import annotations

from dataclasses import dataclass
from urllib.request import urlopen

from dullahan_shared.inference import InferenceHttpError, OpenAICompatibleTextClient
from dullahan_shared.schemas.query import QueryEnvelope


@dataclass(frozen=True)
class PlannerRequest:
    parent_query: QueryEnvelope
    max_breadth: int


@dataclass(frozen=True)
class PlannerResult:
    subqueries: list[str]
    provider: str


class PlannerProvider:
    def plan(self, request: PlannerRequest) -> PlannerResult:
        raise NotImplementedError


class OpenAICompatiblePlannerProvider(PlannerProvider):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        api_mode: str = "completions",
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_mode = api_mode
        self.api_key = api_key

    def plan(self, request: PlannerRequest) -> PlannerResult:
        prompt = self._build_prompt(request)
        try:
            result = OpenAICompatibleTextClient(
                base_url=self.base_url,
                api_mode=self.api_mode,
                api_key=self.api_key,
                timeout_seconds=self.timeout_seconds,
                opener=urlopen,
            ).generate(model=self.model, prompt=prompt, max_tokens=512)
        except InferenceHttpError as exc:
            raise RuntimeError(f"planner provider failed: {exc}") from exc
        return PlannerResult(
            subqueries=self._parse_subqueries(result.text, max_breadth=request.max_breadth),
            provider="openai-compatible-planner",
        )

    def _build_prompt(self, request: PlannerRequest) -> str:
        return "\n".join(
            [
                "Generate only the subqueries needed to answer the parent query.",
                "Each subquery must be specific to the parent query and independently answerable.",
                "Return exactly one subquery per line with no numbering, headings, or reasoning.",
                f"Maximum subqueries: {request.max_breadth}",
                "",
                "Parent query:",
                request.parent_query.query,
            ]
        )

    def _parse_subqueries(self, text: str, *, max_breadth: int) -> list[str]:
        subqueries = []
        for line in text.splitlines():
            cleaned = line.strip().lstrip("-*0123456789. )").strip()
            if cleaned:
                subqueries.append(cleaned)
            if len(subqueries) == max_breadth:
                break
        return subqueries
