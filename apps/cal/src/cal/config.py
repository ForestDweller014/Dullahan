from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class CalConfig(BaseModel):
    repo_root: Path = Field(default_factory=lambda: Path.cwd())
    graph_dir: Path = Path("memory/graph")
    world_state_index_path: Path | None = None
    world_state_backend: Literal["local", "postgres"] = "local"
    postgres_dsn: str | None = None
    postgres_table_name: str = "world_state_documents"
    world_top_k: int = 6
    parent_top_k: int = 4
    token_budget: int = 12000
    inference_base_url: str = "http://127.0.0.1:30000/v1"
    inference_timeout_seconds: float = Field(default=120.0, gt=0)
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dimensions: int = Field(default=1024, gt=0)
    tokenizer_model: str = "Qwen/Qwen3-8B"

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
            inference_base_url=os.getenv(
                "DULLAHAN_INFERENCE_BASE_URL", "http://127.0.0.1:30000/v1"
            ),
            inference_timeout_seconds=float(
                os.getenv("DULLAHAN_INFERENCE_TIMEOUT_SECONDS", "120")
            ),
            embedding_model=os.getenv(
                "DULLAHAN_EMBEDDING_MODEL", "qwen3-embedding:0.6b"
            ),
            embedding_dimensions=int(os.getenv("DULLAHAN_EMBEDDING_DIMENSIONS", "1024")),
            tokenizer_model=os.getenv("DULLAHAN_TOKENIZER_MODEL", "Qwen/Qwen3-8B"),
        )

    @property
    def resolved_graph_dir(self) -> Path:
        if self.graph_dir.is_absolute():
            return self.graph_dir
        return self.repo_root / self.graph_dir

    @property
    def resolved_world_state_index_path(self) -> Path | None:
        if self.world_state_index_path is None:
            return None
        if self.world_state_index_path.is_absolute():
            return self.world_state_index_path
        return self.repo_root / self.world_state_index_path
