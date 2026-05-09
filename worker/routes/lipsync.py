"""POST /lipsync/apply — async lip-sync (chunking + per-chunk model + remerge)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from services.lipsync.service import LipsyncService
from worker.jobs import manager
from worker.schemas import JobAccepted, LipsyncRequest

router = APIRouter(prefix="/lipsync", tags=["lipsync"])


def _run_lipsync(req: LipsyncRequest):
    return LipsyncService(model_id=req.model_id).apply(
        video_path=Path(req.video_path),
        audio_path=Path(req.audio_path),
        out_path=Path(req.out_path),
    )


@router.post("/apply", response_model=JobAccepted, status_code=202)
def apply_lipsync(req: LipsyncRequest) -> JobAccepted:
    job_id = manager.submit(_run_lipsync, req)
    return JobAccepted(job_id=job_id)
