from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from dullahan_inference.plan import ResolvedInferencePlan


class ModelServerError(RuntimeError):
    pass


def _admin_request(
    plan: ResolvedInferencePlan,
    path: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float = 1800,
):
    if not plan.admin_base_url or not plan.admin_token_env:
        raise ModelServerError("inference plan does not target a model-server container")
    token = os.getenv(plan.admin_token_env)
    if not token:
        raise ModelServerError(f"environment variable {plan.admin_token_env} is not set")
    url = f"{plan.admin_base_url}{path}"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={
            "Content-Type": "application/json",
            "X-Admin-Token": token,
        },
        method=method,
    )
    try:
        return urlopen(request, timeout=timeout)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ModelServerError(
            f"model-server request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise ModelServerError(f"model-server request failed: {exc.reason}") from exc


def activate_model_server(plan: ResolvedInferencePlan) -> dict:
    path = f"/admin/models/{quote(plan.model, safe='')}/activate"
    with _admin_request(
        plan,
        path,
        method="POST",
        payload={"extra_args": plan.activation_extra_args},
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def get_model_server_metadata(plan: ResolvedInferencePlan) -> dict:
    path = f"/admin/models/{quote(plan.model, safe='')}"
    with _admin_request(plan, path, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def export_model_server_package(plan: ResolvedInferencePlan, destination: Path) -> Path:
    model = quote(plan.model, safe="")
    mode = quote(plan.model_export_mode, safe="")
    path = f"/admin/models/{model}/archive?mode={mode}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _admin_request(plan, path) as response, destination.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
    return destination
