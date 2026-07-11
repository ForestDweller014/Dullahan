from __future__ import annotations

import re
from threading import RLock

from dullahan_shared.schemas.execution import ExecutionLimits
from dullahan_shared.schemas.query import QueryEnvelope


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


class RecursionGuard:
    def __init__(self, limits: ExecutionLimits) -> None:
        self.limits = limits
        self.seen_query_signatures: set[str] = set()
        self.total_instances = 0
        self._lock = RLock()

    def can_generate_children(self, query: QueryEnvelope) -> bool:
        return query.depth < self.limits.max_depth

    def accept(self, query: QueryEnvelope) -> bool:
        with self._lock:
            if self.total_instances >= self.limits.max_total_agent_instances:
                return False

            signature = self.signature(query.query)
            if signature in self.seen_query_signatures:
                return False

            self.seen_query_signatures.add(signature)
            self.total_instances += 1
            return True

    def signature(self, query: str) -> str:
        return " ".join(token.lower() for token in TOKEN_PATTERN.findall(query))
