from __future__ import annotations

import json
from typing import TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class HttpToolError(RuntimeError):
    pass


def post_json(
    *,
    url: str,
    payload: BaseModel,
    response_model: type[ResponseModel],
    timeout_seconds: float = 30.0,
) -> ResponseModel:
    body = json.dumps(payload.model_dump(mode="json")).encode("utf-8")
    request = Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HttpToolError(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise HttpToolError(f"POST {url} failed: {exc.reason}") from exc

    return response_model.model_validate_json(response_body)
