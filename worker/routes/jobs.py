"""GET/DELETE /jobs/{job_id} — poll status, fetch result, cancel."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.types import MediaAsset
from worker.jobs import manager
from worker.schemas import JobStatus, MediaAssetOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _serialize_result(result) -> MediaAssetOut | None:
    if result is None:
        return None
    if isinstance(result, MediaAsset):
        return MediaAssetOut(
            path=str(result.path), kind=result.kind, meta=result.meta
        )
    # Fallback: trust dict-shaped payloads (future-proof for non-Asset results).
    if isinstance(result, dict) and {"path", "kind"} <= result.keys():
        return MediaAssetOut(**result)
    raise TypeError(f"Cannot serialize job result of type {type(result).__name__}")


@router.get("/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    rec = manager.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")
    return JobStatus(
        job_id=rec.job_id,
        status=rec.status,
        result=_serialize_result(rec.result) if rec.status == "done" else None,
        error=rec.error,
    )


@router.delete("/{job_id}", response_model=JobStatus)
def cancel_job(job_id: str) -> JobStatus:
    rec = manager.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")
    manager.cancel(job_id)
    # Re-read so the returned status reflects the cancel attempt.
    rec = manager.get(job_id)
    return JobStatus(
        job_id=rec.job_id,
        status=rec.status,
        result=_serialize_result(rec.result) if rec.status == "done" else None,
        error=rec.error,
    )
