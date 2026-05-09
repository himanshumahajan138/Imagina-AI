"""POST /tts/synthesize — sync per-segment TTS."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.errors import ImaginaError
from services.tts.service import TTSService
from worker.schemas import MediaAssetOut, TTSRequest

router = APIRouter(prefix="/tts", tags=["tts"])


@router.post("/synthesize", response_model=MediaAssetOut)
def synthesize(req: TTSRequest) -> MediaAssetOut:
    try:
        asset = TTSService(model_id=req.model_id).synthesize(
            text=req.text,
            out_path=Path(req.out_path),
            voice=req.voice,
            speed=req.speed,
            language=req.language,
        )
    except ImaginaError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    return MediaAssetOut(path=str(asset.path), kind=asset.kind, meta=asset.meta)
