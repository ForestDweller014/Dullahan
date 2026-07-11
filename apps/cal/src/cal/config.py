from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class CalConfig(BaseModel):
    repo_root: Path = Field(default_factory=lambda: Path.cwd())
    graph_dir: Path = Path("memory/graph")
    world_state_backend: Literal["local", "postgres"] = "local"
    postgres_dsn: str | None = None
    postgres_table_name: str = "world_state_documents"
    world_top_k: int = 6
    parent_top_k: int = 4
    token_budget: int = 12000

    @classmethod
    def from_env(cls) -> CalConfig:
        repo_root = Path(os.getenv("DULLAHAN_REPO_ROOT", Path.cwd()))
        return cls(
            repo_root=repo_root,
            world_state_backend=os.getenv("WORLD_STATE_BACKEND", "local"),
            postgres_dsn=os.getenv("WORLD_STATE_POSTGRES_DSN"),
            postgres_table_name=os.getenv("WORLD_STATE_POSTGRES_TABLE", "world_state_documents"),
            world_top_k=int(os.getenv("CAL_WORLD_TOP_K", "6")),
            parent_top_k=int(os.getenv("CAL_PARENT_TOP_K", "4")),
            token_budget=int(os.getenv("CAL_TOKEN_BUDGET", "12000")),
        )

    @property
    def resolved_graph_dir(self) -> Path:
        if self.graph_dir.is_absolute():
            return self.graph_dir
        return self.repo_root / self.graph_dir
