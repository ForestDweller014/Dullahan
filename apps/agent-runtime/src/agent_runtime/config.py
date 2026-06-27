from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from dullahan_shared.schemas.execution import ExecutionLimits


class AgentRuntimeConfig(BaseModel):
    repo_root: Path = Field(default_factory=lambda: Path.cwd())
    limits: ExecutionLimits = Field(default_factory=ExecutionLimits)
    max_sibling_concurrency: int = Field(default=8, ge=1)
    planner_provider: str = "deterministic"
    planner_model: str = "local-planner"
    planner_base_url: str = "http://127.0.0.1:30000/v1"
    planner_timeout_seconds: float = Field(default=30.0, gt=0)

    @classmethod
    def from_files(cls, repo_root: Path) -> AgentRuntimeConfig:
        recursion_path = repo_root / "configs" / "recursion.yaml"
        with recursion_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        return cls(
            repo_root=repo_root,
            limits=ExecutionLimits.model_validate(data),
            max_sibling_concurrency=data.get("max_sibling_concurrency", 8),
        )
