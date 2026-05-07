"""Imagina worker — FastAPI entrypoint.

Run locally:
    uvicorn worker.main:app --reload --port 8005

Health-check:
    curl localhost:8005/health
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from core.errors import ImaginaError
from core.logger import logger
from worker.routes import health, images, jobs, lipsync, models, scripts, tts, videos

load_dotenv()


# Worker-owned scratch dir for any output files where the caller didn't
# specify `out_path`. We don't auto-clean here; a TTL janitor lands in
# Phase 2 along with the async job manager.
WORKER_TMP = Path(os.getenv("IMAGINA_WORKER_TMP", "/tmp/imagina"))
WORKER_TMP.mkdir(parents=True, exist_ok=True)


def _preload_enabled() -> bool:
    return os.getenv("IMAGINA_PRELOAD_LLM", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _warmup_llm() -> None:
    """Background-thread warmup for the auto-picked LLM.

    Skipped when the auto-pick is an API/cloud_oss tier (no local weights
    to load) or when IMAGINA_PRELOAD_LLM is set to a falsy value. Failures
    are logged but never raise — worker stays up either way.
    """
    try:
        from services.llm.service import LLMService

        svc = LLMService()
        if svc.cfg.get("tier") != "local":
            logger.info(
                f"[worker] LLM preload skipped — auto-pick is {svc.model_id} "
                f"(tier={svc.cfg.get('tier')}, no local weights)"
            )
            return

        warmup = getattr(svc._backend, "warmup", None)
        if warmup is None:
            logger.info(
                f"[worker] LLM preload skipped — backend {svc._backend.name} "
                "has no warmup() hook"
            )
            return

        logger.info(f"[worker] preloading LLM: {svc.model_id} (this may take ~30s)")
        warmup()
        logger.info(f"[worker] LLM preloaded: {svc.model_id}")
    except ImaginaError as e:
        logger.warning(f"[worker] LLM preload skipped: {e}")
    except Exception as e:
        logger.exception(f"[worker] LLM preload crashed: {e}")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Imagina Worker",
        description="Holds all model calls so Streamlit stays a thin client.",
        version="0.1.0",
    )
    app.include_router(health.router)
    app.include_router(scripts.router)
    app.include_router(images.router)
    app.include_router(tts.router)
    app.include_router(videos.router)
    app.include_router(lipsync.router)
    app.include_router(jobs.router)
    app.include_router(models.router)

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info(f"[worker] startup | tmp={WORKER_TMP}")
        if _preload_enabled():
            # Daemon thread so it doesn't block uvicorn's startup; the
            # /health endpoint is responsive while weights load.
            threading.Thread(
                target=_warmup_llm, daemon=True, name="llm-warmup"
            ).start()

    return app


app = create_app()


def main() -> None:
    """Module-level entrypoint for `python -m worker.main`."""
    import uvicorn

    port = int(os.getenv("IMAGINA_WORKER_PORT", "8005"))
    uvicorn.run("worker.main:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()
