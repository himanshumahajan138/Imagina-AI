"""Pydantic request/response shapes for the worker HTTP API.

Mirror the existing `services/<modality>/service.py` signatures so the
client lib (`core/worker_client.py`) and pipelines can swap between
direct calls and HTTP calls with no shape translation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ─── Common ──────────────────────────────────────────────────────────


class MediaAssetOut(BaseModel):
    """Wire-format twin of `core.types.MediaAsset`."""

    path: str
    kind: Literal["image", "audio", "video"]
    meta: dict[str, Any] = Field(default_factory=dict)


# ─── Script (LLM) ────────────────────────────────────────────────────


class ScriptRequest(BaseModel):
    theme: str
    duration: int
    language: str
    model_id: str | None = None
    model_type: str | None = None  # legacy hint forwarded to backends


class ScriptBlockOut(BaseModel):
    script: str
    scene: str
    video_scene: str
    start_time: str
    end_time: str


class ScriptResponse(BaseModel):
    blocks: list[ScriptBlockOut]
    theme: str | None = None
    duration: int | None = None
    language: str | None = None


# ─── Image ───────────────────────────────────────────────────────────


class ImageRequest(BaseModel):
    prompt: str
    out_path: str
    dimension: str
    reference_images: list[str] = Field(default_factory=list)
    script: str = ""
    video_scene: str = ""
    reference_text: str | None = None
    old_image: str | None = None
    previous_response_id: str | None = None
    model_id: str | None = None


# ─── Video ───────────────────────────────────────────────────────────


class VideoRequest(BaseModel):
    prompt: str
    out_path: str
    dimension: str
    duration: float
    seed_image: str | None = None
    model_id: str | None = None


# ─── TTS ─────────────────────────────────────────────────────────────


class TTSRequest(BaseModel):
    text: str
    out_path: str
    voice: str
    speed: float = 1.0
    language: str = "a"
    model_id: str | None = None


# ─── Lipsync ─────────────────────────────────────────────────────────


class LipsyncRequest(BaseModel):
    video_path: str
    audio_path: str
    out_path: str
    model_id: str | None = None


# ─── Jobs (async endpoints) ──────────────────────────────────────────


class JobAccepted(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error", "cancelled"]
    result: MediaAssetOut | None = None
    error: str | None = None


# ─── Health / introspection ──────────────────────────────────────────


class ModalityPick(BaseModel):
    modality: str
    auto_picked: str | None = None  # model_id chosen by registry, if any
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    picks: list[ModalityPick] = Field(default_factory=list)
