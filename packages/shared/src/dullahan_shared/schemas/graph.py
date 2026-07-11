from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    FILE = "file"
    CLASS = "class"
    CONCEPT = "concept"
    IMAGE = "image"
    SERVICE = "service"
    EXPERT = "expert"


class EdgeType(StrEnum):
    IMPORTS = "imports"
    REFERENCES = "references"
    CONNECTS_TO = "connects_to"
    OWNS = "owns"
    SEMANTICALLY_RELATED = "semantically_related"


class GraphNode(BaseModel):
    id: str
    type: NodeType
    title: str = Field(min_length=1)
    document_paths: list[str] = Field(default_factory=list)
    cluster_id: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = Field(default=1.0, ge=0.0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
