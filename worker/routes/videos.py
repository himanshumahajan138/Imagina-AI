"""POST /videos/generate — async per-scene video generation."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from services.video.service import VideoService
from worker.jobs import manager
from worker.schemas import JobAccepted, VideoRequest

router = APIRouter(prefix="/videos", tags=["videos"])


def _run_video(req: VideoRequest):
    """Worker-thread body. Returns the raw `MediaAsset` from the service."""
    return VideoService(model_id=req.model_id).generate_video(
        prompt=req.prompt,
        out_path=Path(req.out_path),
        dimension=req.dimension,
        duration=req.duration,
        seed_image=Path(req.seed_image) if req.seed_image else None,
    )


@router.post("/generate", response_model=JobAccepted, status_code=202)
def generate_video(req: VideoRequest) -> JobAccepted:
    job_id = manager.submit(_run_video, req)
    return JobAccepted(job_id=job_id)
