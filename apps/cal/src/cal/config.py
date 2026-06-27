from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class CalConfig(BaseModel):
    repo_root: Path = Field(default_factory=lambda: Path.cwd())
    graph_dir: Path = Path("memory/graph")
    world_top_k: int = 6
    parent_top_k: int = 4
    token_budget: int = 12000

    @classmethod
    def from_env(cls) -> CalConfig:
        repo_root = Path(os.getenv("DULLAHAN_REPO_ROOT", Path.cwd()))
        return cls(repo_root=repo_root)

    @property
    def resolved_graph_dir(self) -> Path:
        if self.graph_dir.is_absolute():
            return self.graph_dir
        return self.repo_root / self.graph_dir
