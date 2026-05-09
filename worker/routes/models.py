"""Active model-cache management.

`POST /models/evict` lets pipelines declare phase boundaries — e.g. "image
gen done, free SDXL before video starts" — so on a 16 GB box we don't
thrash between MLX LLM (~6 GB) and SDXL (~7 GB).

`GET /models/resident` is a debug helper: returns the current cache
snapshot so you can see what the worker is holding.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from core.model_manager import get_manager

router = APIRouter(prefix="/models", tags=["models"])


class EvictRequest(BaseModel):
    modality: str | None = None  # "llm" | "image" | "tts" | "video" | "lipsync"
    key: str | None = None       # exact cache key (e.g. "diffusers::stabilityai/sdxl-turbo")
    all: bool = False


class EvictResponse(BaseModel):
    evicted: int
    resident: dict[str, float]


@router.post("/evict", response_model=EvictResponse)
def evict(req: EvictRequest) -> EvictResponse:
    mgr = get_manager()
    if req.all:
        count = mgr.evict_all()
    elif req.key:
        count = 1 if mgr.evict(req.key) else 0
    elif req.modality:
        count = mgr.evict_modality(req.modality)
    else:
        count = 0
    return EvictResponse(evicted=count, resident=mgr.resident())


@router.get("/resident", response_model=dict[str, float])
def resident() -> dict[str, float]:
    return get_manager().resident()
