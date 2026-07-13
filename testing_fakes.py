from __future__ import annotations

import re


class KeywordEmbeddingModel:
    """Explicit unit-test fake; real semantic quality is covered by inference tests."""

    model_id = "test-keyword-embedding"
    vocabulary = (
        "cal",
        "context",
        "world",
        "edl",
        "expert",
        "dispatch",
        "routing",
        "graph",
        "memory",
        "knowledge",
        "agent",
        "runtime",
        "execution",
        "token",
        "portfolio",
        "risk",
        "duration",
        "curve",
    )
    dimensions = len(vocabulary)

    def embed(self, text: str) -> list[float]:
        terms = set(re.findall(r"[a-z0-9_]+", text.lower()))
        return [1.0 if term in terms else 0.0 for term in self.vocabulary]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class WhitespaceTokenCounter:
    """Explicit unit-test fake for deterministic budget assertions."""

    model_id = "test-whitespace-tokenizer"

    def count(self, text: str) -> int:
        return len(text.split())
