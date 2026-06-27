from __future__ import annotations

from fastapi import APIRouter

from edl.api.schemas import (
    BatchDispatchRequest,
    BatchDispatchResponse,
    DispatchRequest,
    DispatchResponse,
)
from edl.config import EdlConfig
from edl.service import ExpertDispatchService


router = APIRouter()
service = ExpertDispatchService.from_config(EdlConfig.from_env())


@router.post("/dispatch", response_model=DispatchResponse)
def dispatch(request: DispatchRequest) -> DispatchResponse:
    return service.dispatch(request)


@router.post("/dispatch/batch", response_model=BatchDispatchResponse)
def dispatch_batch(request: BatchDispatchRequest) -> BatchDispatchResponse:
    return service.dispatch_batch(request)
