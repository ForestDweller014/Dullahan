from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dullahan_shared.inference_config import (
    embedding_model,
    generation_model,
    inference_api_key,
    inference_base_url,
    inference_provider,
)
from pydantic import BaseModel, Field, SecretStr


class EdlConfig(BaseModel):
    repo_root: Path = Field(default_factory=lambda: Path.cwd())
    experts_path: Path = Path("memory/graph/experts.yaml")
    min_score_threshold: float = 0.0
    max_dispatch_concurrency: int = Field(default=16, ge=1)
    model_provider: Literal["http", "openai"] = "http"
    model_base_url: str = "http://127.0.0.1:30000/v1"
    model_api_key: SecretStr | None = None
    model_override: str | None = None
    model_timeout_seconds: float = Field(default=30.0, gt=0)
    model_max_tokens: int = Field(default=512, ge=1)
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dimensions: int = Field(default=1024, gt=0)

    @classmethod
    def from_env(cls) -> EdlConfig:
        repo_root = Path(os.getenv("DULLAHAN_REPO_ROOT", Path.cwd()))
        provider = inference_provider("EDL_MODEL_PROVIDER")
        return cls(
            repo_root=repo_root,
            max_dispatch_concurrency=int(os.getenv("EDL_MAX_DISPATCH_CONCURRENCY", "16")),
            model_provider=provider,
            model_base_url=inference_base_url(provider, "EDL_MODEL_BASE_URL"),
            model_api_key=inference_api_key(provider, "EDL_MODEL_API_KEY"),
            model_override=(
                generation_model(
                    provider,
                    specific_name="EDL_MODEL",
                    local_default="",
                )
                if provider == "openai"
                else os.getenv("EDL_MODEL")
            ),
            model_timeout_seconds=float(os.getenv("EDL_MODEL_TIMEOUT_SECONDS", "30")),
            model_max_tokens=int(os.getenv("EDL_MODEL_MAX_TOKENS", "512")),
            embedding_model=embedding_model(provider),
            embedding_dimensions=int(os.getenv("DULLAHAN_EMBEDDING_DIMENSIONS", "1024")),
        )

    @property
    def resolved_experts_path(self) -> Path:
        if self.experts_path.is_absolute():
            return self.experts_path
        return self.repo_root / self.experts_path
