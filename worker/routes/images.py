"""POST /images/generate — async cinematic still.

First-run weight downloads (SDXL-Turbo ≈ 7 GB) plus inference can take
many minutes, well past any reasonable sync HTTP timeout. We submit the
work to the same `JobManager` the video route uses and return a
`job_id`; the client polls `/jobs/{id}` to completion.

Subsequent calls reuse the cached pipeline (via `core.model_manager`)
and complete in ~6-12s on M2.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from services.image.service import ImageService
from worker.jobs import manager
from worker.schemas import ImageRequest, JobAccepted

router = APIRouter(prefix="/images", tags=["images"])


def _run_image(req: ImageRequest):
    """Worker-thread body. Returns the raw `MediaAsset` from the service."""
    return ImageService(model_id=req.model_id).generate_image(
        prompt=req.prompt,
        out_path=Path(req.out_path),
        dimension=req.dimension,
        reference_images=[Path(p) for p in req.reference_images] or None,
        script=req.script,
        video_scene=req.video_scene,
        reference_text=req.reference_text,
        old_image=req.old_image,
        previous_response_id=req.previous_response_id,
    )


@router.post("/generate", response_model=JobAccepted, status_code=202)
def generate_image(req: ImageRequest) -> JobAccepted:
    job_id = manager.submit(_run_image, req)
    return JobAccepted(job_id=job_id)
