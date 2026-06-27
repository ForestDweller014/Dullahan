from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from math import sqrt
from collections.abc import Callable
from typing import Generic, TypeVar


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
T = TypeVar("T")


@dataclass(frozen=True)
class RankedItem(Generic[T]):
    item: T
    score: float


class LexicalRetriever:
    def rank(
        self,
        query: str,
        items: list[T],
        *,
        text_of: Callable[[T], str],
        id_of: Callable[[T], str],
        top_k: int,
    ) -> list[RankedItem[T]]:
        if top_k <= 0:
            return []

        query_vector = self._vectorize(query)
        if not query_vector:
            return []

        ranked: list[RankedItem[T]] = []
        for item in items:
            score = self._cosine(query_vector, self._vectorize(text_of(item)))
            if score > 0:
                ranked.append(RankedItem(item=item, score=score))

        return sorted(ranked, key=lambda ranked_item: (-ranked_item.score, id_of(ranked_item.item)))[
            :top_k
        ]

    def _vectorize(self, text: str) -> Counter[str]:
        return Counter(token.lower() for token in TOKEN_PATTERN.findall(text))

    def _cosine(self, left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0

        dot = sum(left[token] * right[token] for token in left.keys() & right.keys())
        if dot == 0:
            return 0.0

        left_norm = sqrt(sum(value * value for value in left.values()))
        right_norm = sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm)
