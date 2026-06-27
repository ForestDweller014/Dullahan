from __future__ import annotations

from fastapi import APIRouter

from cal.api.schemas import (
    AugmentContextRequest,
    AugmentContextResponse,
    BatchAugmentContextRequest,
    BatchAugmentContextResponse,
)
from cal.config import CalConfig
from cal.service import ContextAugmentationService


router = APIRouter()
service = ContextAugmentationService.from_config(CalConfig.from_env())


@router.post("/augment", response_model=AugmentContextResponse)
def augment_context(request: AugmentContextRequest) -> AugmentContextResponse:
    return service.augment(request)


@router.post("/augment/batch", response_model=BatchAugmentContextResponse)
def augment_context_batch(request: BatchAugmentContextRequest) -> BatchAugmentContextResponse:
    return service.augment_batch(request)
