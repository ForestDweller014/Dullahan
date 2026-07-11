from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str) -> str:
    """Create a stable, readable identifier for runtime envelopes."""
    normalized = prefix.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("id prefix cannot be empty")
    return f"{normalized}:{uuid4().hex}"
