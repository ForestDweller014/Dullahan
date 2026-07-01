from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ContextSource(StrEnum):
    PARENT = "parent"
    WORLD_STATE = "world_state"
    GRAPH_NODE = "graph_node"
    GRAPH_CLUSTER = "graph_cluster"


class ContextDocument(BaseModel):
    id: str
    source: ContextSource
    text: str = Field(min_length=1)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ContextBundle(BaseModel):
    query_id: str
    documents: list[ContextDocument] = Field(default_factory=list)
    token_budget: int | None = Field(default=None, gt=0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(document.text for document in self.documents)
