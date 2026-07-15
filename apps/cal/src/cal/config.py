from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dullahan_shared.inference_config import (
    DEFAULT_LOCAL_TOKENIZER_MODEL,
    embedding_model,
    inference_api_key,
    inference_base_url,
    inference_provider,
    tokenizer_api_key,
    tokenizer_base_url,
)
from pydantic import BaseModel, Field, SecretStr


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
    inference_provider: Literal["http", "openai"] = "http"
    inference_base_url: str = "http://127.0.0.1:30000/v1"
    inference_api_key: SecretStr | None = None
    inference_timeout_seconds: float = Field(default=120.0, gt=0)
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dimensions: int = Field(default=1024, gt=0)
    tokenizer_model: str = "Qwen/Qwen3-8B"
    tokenizer_base_url: str = "http://127.0.0.1:30000/v1"
    tokenizer_api_key: SecretStr | None = None

    @classmethod
    def from_env(cls) -> CalConfig:
        repo_root = Path(os.getenv("DULLAHAN_REPO_ROOT", Path.cwd()))
        provider = inference_provider()
        selected_inference_base_url = inference_base_url(provider)
        selected_inference_api_key = inference_api_key(provider)
        return cls(
            repo_root=repo_root,
            world_state_backend=os.getenv("WORLD_STATE_BACKEND", "local"),
            postgres_dsn=os.getenv("WORLD_STATE_POSTGRES_DSN"),
            postgres_table_name=os.getenv("WORLD_STATE_POSTGRES_TABLE", "world_state_documents"),
            world_top_k=int(os.getenv("CAL_WORLD_TOP_K", "6")),
            parent_top_k=int(os.getenv("CAL_PARENT_TOP_K", "4")),
            token_budget=int(os.getenv("CAL_TOKEN_BUDGET", "12000")),
            inference_provider=provider,
            inference_base_url=selected_inference_base_url,
            inference_api_key=selected_inference_api_key,
            inference_timeout_seconds=float(
                os.getenv("DULLAHAN_INFERENCE_TIMEOUT_SECONDS", "120")
            ),
            embedding_model=embedding_model(provider),
            embedding_dimensions=int(os.getenv("DULLAHAN_EMBEDDING_DIMENSIONS", "1024")),
            tokenizer_model=os.getenv(
                "DULLAHAN_TOKENIZER_MODEL",
                DEFAULT_LOCAL_TOKENIZER_MODEL,
            ),
            tokenizer_base_url=tokenizer_base_url(
                provider,
                selected_inference_base_url=selected_inference_base_url,
            ),
            tokenizer_api_key=tokenizer_api_key(
                provider,
                selected_inference_api_key=selected_inference_api_key,
            ),
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
