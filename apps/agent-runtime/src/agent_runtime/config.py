from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dullahan_shared.inference_config import (
    generation_model,
    inference_api_key,
    inference_base_url,
    inference_provider,
)
from dullahan_shared.schemas.execution import ExecutionLimits
from pydantic import BaseModel, Field, SecretStr


class AgentRuntimeConfig(BaseModel):
    repo_root: Path = Field(default_factory=lambda: Path.cwd())
    limits: ExecutionLimits = Field(default_factory=ExecutionLimits)
    max_sibling_concurrency: int = Field(default=8, ge=1)
    planner_provider: Literal["http", "openai"] = "http"
    planner_model: str = "local-planner"
    planner_base_url: str = "http://127.0.0.1:30000/v1"
    planner_api_key: SecretStr | None = None
    planner_timeout_seconds: float = Field(default=30.0, gt=0)
    synthesis_provider: Literal["http", "openai"] = "http"
    synthesis_model: str = "local-planner"
    synthesis_base_url: str = "http://127.0.0.1:30000/v1"
    synthesis_api_key: SecretStr | None = None
    synthesis_timeout_seconds: float = Field(default=60.0, gt=0)
    synthesis_max_tokens: int = Field(default=1024, ge=1)

    @classmethod
    def from_files(cls, repo_root: Path) -> AgentRuntimeConfig:
        recursion_path = repo_root / "configs" / "recursion.yaml"
        with recursion_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        planner_provider = inference_provider("AGENT_PLANNER_PROVIDER")
        synthesis_provider = inference_provider("AGENT_SYNTHESIS_PROVIDER")
        planner_model = generation_model(
            planner_provider,
            specific_name="AGENT_PLANNER_MODEL",
            local_default="local-planner",
        )
        planner_base_url = inference_base_url(planner_provider, "AGENT_PLANNER_BASE_URL")
        planner_api_key = inference_api_key(planner_provider, "AGENT_PLANNER_API_KEY")
        synthesis_model = os.getenv("AGENT_SYNTHESIS_MODEL")
        if not synthesis_model:
            synthesis_model = (
                planner_model
                if synthesis_provider == planner_provider
                else generation_model(
                    synthesis_provider,
                    local_default="local-planner",
                )
            )
        synthesis_base_url = os.getenv("AGENT_SYNTHESIS_BASE_URL")
        if not synthesis_base_url:
            synthesis_base_url = (
                planner_base_url
                if synthesis_provider == planner_provider
                else inference_base_url(synthesis_provider)
            )
        synthesis_api_key = os.getenv("AGENT_SYNTHESIS_API_KEY")
        if not synthesis_api_key:
            synthesis_api_key = (
                planner_api_key
                if synthesis_provider == planner_provider
                else inference_api_key(synthesis_provider)
            )
        return cls(
            repo_root=repo_root,
            limits=ExecutionLimits.model_validate(data),
            max_sibling_concurrency=data.get("max_sibling_concurrency", 8),
            planner_provider=planner_provider,
            planner_model=planner_model,
            planner_base_url=planner_base_url,
            planner_api_key=planner_api_key,
            planner_timeout_seconds=float(os.getenv("AGENT_PLANNER_TIMEOUT_SECONDS", "30")),
            synthesis_provider=synthesis_provider,
            synthesis_model=synthesis_model,
            synthesis_base_url=synthesis_base_url,
            synthesis_api_key=synthesis_api_key,
            synthesis_timeout_seconds=float(os.getenv("AGENT_SYNTHESIS_TIMEOUT_SECONDS", "60")),
            synthesis_max_tokens=int(os.getenv("AGENT_SYNTHESIS_MAX_TOKENS", "1024")),
        )
