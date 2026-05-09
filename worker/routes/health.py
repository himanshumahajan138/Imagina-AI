"""Liveness + registry introspection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from core.errors import ImaginaError
from core.registry import list_models, pick_model
from worker.schemas import HealthResponse, ModalityPick

router = APIRouter()

_MODALITIES = ("llm", "image", "video", "lipsync", "tts")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    picks: list[ModalityPick] = []
    for modality in _MODALITIES:
        try:
            mid, _ = pick_model(modality)
            picks.append(ModalityPick(modality=modality, auto_picked=mid))
        except ImaginaError as e:
            picks.append(ModalityPick(modality=modality, error=str(e)))
    return HealthResponse(picks=picks)


@router.get("/models")
def models() -> dict[str, Any]:
    """Full registry dump — handy for the sidebar tier picker."""
    return {modality: list_models(modality) for modality in _MODALITIES}
