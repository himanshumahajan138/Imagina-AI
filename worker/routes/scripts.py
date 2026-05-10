"""POST /scripts/generate — sync LLM script generation."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from core.errors import ImaginaError
from services.llm.service import LLMService
from worker.schemas import ScriptBlockOut, ScriptRequest, ScriptResponse

router = APIRouter(prefix="/scripts", tags=["scripts"])


@router.post("/generate", response_model=ScriptResponse)
def generate_script(req: ScriptRequest) -> ScriptResponse:
    try:
        svc = LLMService(model_id=req.model_id)
        # LLMService.generate_script returns a DataFrame; we want raw blocks.
        # Reach into the backend so we can return the structured Script.
        kwargs = {}
        if req.model_type:
            kwargs["model_type"] = req.model_type
        if req.seconds:
            kwargs["seconds"] = req.seconds
        script = svc._backend.generate_script(
            theme=req.theme,
            duration=req.duration,
            language=req.language,
            **kwargs,
        )
    except ImaginaError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    return ScriptResponse(
        blocks=[ScriptBlockOut(**asdict(b)) for b in script.blocks],
        theme=script.theme,
        duration=script.duration,
        language=script.language,
    )
