from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dullahan_shared.schemas.expert import ExpertProfile


class ExpertRegistry:
    def __init__(self, repo_root: Path, experts_path: Path) -> None:
        self.repo_root = repo_root
        self.experts_path = experts_path

    def load(self) -> list[ExpertProfile]:
        data = self._read_yaml()
        profiles: list[ExpertProfile] = []

        for item in data.get("experts", []):
            role_context_path = item["role_context_path"]
            role_context = (self.repo_root / role_context_path).read_text(encoding="utf-8")
            profiles.append(
                ExpertProfile(
                    id=item["id"],
                    cluster_id=item["cluster_id"],
                    role_context=role_context,
                    model=item["model"],
                    max_concurrency=item.get("max_concurrency", 1),
                    metadata={
                        **item.get("metadata", {}),
                        "role_context_path": role_context_path,
                    },
                )
            )

        return profiles

    def _read_yaml(self) -> dict[str, Any]:
        with self.experts_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {"experts": []}
        if not isinstance(data, dict):
            raise ValueError(f"expected YAML mapping in {self.experts_path}")
        return data
