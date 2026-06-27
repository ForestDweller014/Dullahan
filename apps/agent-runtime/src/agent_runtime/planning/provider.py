from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

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


class DeterministicPlannerProvider(PlannerProvider):
    def plan(self, request: PlannerRequest) -> PlannerResult:
        candidates = [
            "What context should CAL retrieve?",
            "Which expert should EDL select?",
            "What knowledge graph concepts are relevant?",
        ]
        return PlannerResult(
            subqueries=candidates[: request.max_breadth],
            provider="deterministic-planner",
        )


class OpenAICompatiblePlannerProvider(PlannerProvider):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def plan(self, request: PlannerRequest) -> PlannerResult:
        prompt = self._build_prompt(request)
        http_request = UrlRequest(
            url=f"{self.base_url}/completions",
            data=json.dumps(
                {
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": 512,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"planner provider failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"planner provider request failed: {exc.reason}") from exc

        data = json.loads(response_body)
        text = self._extract_text(data)
        return PlannerResult(
            subqueries=self._parse_subqueries(text, max_breadth=request.max_breadth),
            provider="openai-compatible-planner",
        )

    def _build_prompt(self, request: PlannerRequest) -> str:
        return "\n".join(
            [
                "Generate only the subqueries needed to answer the parent query.",
                "Return one subquery per line. Do not include reasoning.",
                f"Maximum subqueries: {request.max_breadth}",
                "",
                "Parent query:",
                request.parent_query.query,
            ]
        )

    def _extract_text(self, data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        first_choice = choices[0]
        if "text" in first_choice:
            return str(first_choice["text"])
        message = first_choice.get("message")
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return ""

    def _parse_subqueries(self, text: str, *, max_breadth: int) -> list[str]:
        subqueries = []
        for line in text.splitlines():
            cleaned = line.strip().lstrip("-*0123456789. )").strip()
            if cleaned:
                subqueries.append(cleaned)
            if len(subqueries) == max_breadth:
                break
        return subqueries
