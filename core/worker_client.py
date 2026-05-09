"""HTTP client for the Imagina worker daemon.

The Streamlit pipelines import `worker` from this module instead of the
`*Service` facades directly. Method shapes mirror the service-level API
so migration is mechanical:

    LLMService().generate_script(...)        →  worker.generate_script(...)
    ImageService().generate_image(...)       →  worker.generate_image(...)
    VideoService().generate_video(...)       →  worker.generate_video(...)
    TTSService().synthesize(...)             →  worker.synthesize(...)
    LipsyncService().apply(...)              →  worker.apply_lipsync(...)

Long-running endpoints (video, lipsync) poll `/jobs/{id}` internally and
return the final `MediaAsset` so callers don't have to know about jobs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pandas as pd

from core.errors import BackendUnavailable, GenerationFailed
from core.logger import logger
from core.types import MediaAsset, Script, ScriptBlock, Tier
from core.worker_config import (
    DEFAULT_JOB_TIMEOUT_S,
    DEFAULT_POLL_INTERVAL_S,
    SYNC_TIMEOUT_S,
    WORKER_URL,
)


class WorkerClient:
    """Thin httpx wrapper around the worker HTTP API."""

    def __init__(
        self,
        base_url: str = WORKER_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        # `transport` lets tests inject `httpx.WSGITransport(app=fastapi_app)`
        # so the client exercises real FastAPI routing without sockets.
        self._client = httpx.Client(
            base_url=self._base,
            timeout=SYNC_TIMEOUT_S,
            transport=transport,
        )

    # ─── Internals ────────────────────────────────────────────────

    def _post(self, path: str, **payload: Any) -> dict[str, Any]:
        body = {k: v for k, v in payload.items() if v is not None}
        try:
            r = self._client.post(path, json=body)
        except httpx.RequestError as e:
            raise BackendUnavailable(f"Worker unreachable at {self._base}: {e}") from e
        if r.status_code >= 400:
            self._raise_for_status(r)
        return r.json()

    def _get(self, path: str) -> dict[str, Any]:
        try:
            r = self._client.get(path)
        except httpx.RequestError as e:
            raise BackendUnavailable(f"Worker unreachable at {self._base}: {e}") from e
        if r.status_code >= 400:
            self._raise_for_status(r)
        return r.json()

    @staticmethod
    def _raise_for_status(r: httpx.Response) -> None:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        if r.status_code == 502:
            # Worker rebroadcasts ImaginaError → pipeline expects GenerationFailed.
            raise GenerationFailed(detail)
        if r.status_code == 503:
            raise BackendUnavailable(detail)
        raise GenerationFailed(f"Worker {r.status_code}: {detail}")

    @staticmethod
    def _to_asset(payload: dict[str, Any]) -> MediaAsset:
        return MediaAsset(
            path=Path(payload["path"]),
            kind=payload["kind"],
            meta=payload.get("meta", {}) or {},
        )

    # ─── Health / introspection ───────────────────────────────────

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def models(self) -> dict[str, Any]:
        return self._get("/models")

    def is_alive(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False

    def evict_models(
        self,
        modality: str | None = None,
        key: str | None = None,
        all: bool = False,
    ) -> dict[str, Any]:
        """Best-effort hint to the worker to drop loaded weights.

        Pipelines call this at phase boundaries (e.g. after script gen,
        before image gen) so the next phase has memory headroom. Failures
        are logged and swallowed — eviction is an optimisation, never a
        correctness-critical step.
        """
        body: dict[str, Any] = {}
        if modality is not None:
            body["modality"] = modality
        if key is not None:
            body["key"] = key
        if all:
            body["all"] = True
        try:
            return self._post("/models/evict", **body)
        except Exception as e:
            logger.warning(f"[worker_client] evict_models hint failed: {e}")
            return {"evicted": 0, "resident": {}}

    # ─── Sync endpoints ───────────────────────────────────────────

    def generate_script(
        self,
        theme: str,
        duration: int,
        language: str,
        model_id: str | None = None,
        model_type: str | None = None,
    ) -> pd.DataFrame:
        """Mirrors `LLMService.generate_script` — returns a DataFrame of beats."""
        data = self._post(
            "/scripts/generate",
            theme=theme,
            duration=duration,
            language=language,
            model_id=model_id,
            model_type=model_type,
        )
        return pd.json_normalize(data["blocks"])

    def generate_script_raw(
        self,
        theme: str,
        duration: int,
        language: str,
        model_id: str | None = None,
        model_type: str | None = None,
    ) -> Script:
        """Same as `generate_script` but returns the typed `Script` dataclass."""
        data = self._post(
            "/scripts/generate",
            theme=theme,
            duration=duration,
            language=language,
            model_id=model_id,
            model_type=model_type,
        )
        return Script(
            blocks=[ScriptBlock(**b) for b in data["blocks"]],
            theme=data.get("theme"),
            duration=data.get("duration"),
            language=data.get("language"),
        )

    def generate_image(
        self,
        prompt: str,
        out_path: Path | str,
        dimension: str,
        reference_images: list[Path | str] | None = None,
        model_id: str | None = None,
        poll_interval: float = 2.0,
        timeout: float = DEFAULT_JOB_TIMEOUT_S,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> MediaAsset:
        # Async on the worker side — first-run model downloads can take
        # several minutes for diffusers backends. Poll quickly (2s) since
        # cached calls complete in ~6-12s and tighter polling gives more
        # responsive UI feedback.
        accepted = self._post(
            "/images/generate",
            prompt=prompt,
            out_path=str(out_path),
            dimension=dimension,
            reference_images=[str(p) for p in (reference_images or [])],
            script=kwargs.get("script", ""),
            video_scene=kwargs.get("video_scene", ""),
            reference_text=kwargs.get("reference_text"),
            old_image=str(kwargs["old_image"]) if kwargs.get("old_image") else None,
            previous_response_id=kwargs.get("previous_response_id"),
            model_id=model_id,
        )
        return self._await_job(accepted["job_id"], poll_interval, timeout, on_progress)

    def synthesize(
        self,
        text: str,
        out_path: Path | str,
        voice: str,
        speed: float = 1.0,
        language: str = "a",
        model_id: str | None = None,
    ) -> MediaAsset:
        data = self._post(
            "/tts/synthesize",
            text=text,
            out_path=str(out_path),
            voice=voice,
            speed=speed,
            language=language,
            model_id=model_id,
        )
        return self._to_asset(data)

    # ─── Async endpoints (poll under the hood) ────────────────────

    def generate_video(
        self,
        prompt: str,
        out_path: Path | str,
        dimension: str,
        duration: float,
        seed_image: Path | str | None = None,
        model_id: str | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
        timeout: float = DEFAULT_JOB_TIMEOUT_S,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> MediaAsset:
        accepted = self._post(
            "/videos/generate",
            prompt=prompt,
            out_path=str(out_path),
            dimension=dimension,
            duration=duration,
            seed_image=str(seed_image) if seed_image else None,
            model_id=model_id,
        )
        return self._await_job(accepted["job_id"], poll_interval, timeout, on_progress)

    def apply_lipsync(
        self,
        video_path: Path | str,
        audio_path: Path | str,
        out_path: Path | str,
        model_id: str | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
        timeout: float = DEFAULT_JOB_TIMEOUT_S,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> MediaAsset:
        accepted = self._post(
            "/lipsync/apply",
            video_path=str(video_path),
            audio_path=str(audio_path),
            out_path=str(out_path),
            model_id=model_id,
        )
        return self._await_job(accepted["job_id"], poll_interval, timeout, on_progress)

    # ─── Job polling ──────────────────────────────────────────────

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        try:
            r = self._client.delete(f"/jobs/{job_id}")
        except httpx.RequestError as e:
            raise BackendUnavailable(f"Worker unreachable: {e}") from e
        if r.status_code >= 400:
            self._raise_for_status(r)
        return r.json()

    def _await_job(
        self,
        job_id: str,
        poll_interval: float,
        timeout: float,
        on_progress: Callable[[dict[str, Any]], None] | None,
    ) -> MediaAsset:
        deadline = time.monotonic() + timeout
        last_status: str | None = None
        while True:
            status = self.get_job(job_id)
            if on_progress:
                try:
                    on_progress(status)
                except Exception:
                    logger.exception("on_progress callback failed")
            if status["status"] != last_status:
                logger.info(f"[job {job_id}] {last_status} → {status['status']}")
                last_status = status["status"]

            if status["status"] == "done":
                return self._to_asset(status["result"])
            if status["status"] == "error":
                raise GenerationFailed(status.get("error") or "worker job failed")
            if status["status"] == "cancelled":
                raise GenerationFailed(f"job {job_id} was cancelled")

            if time.monotonic() > deadline:
                self.cancel_job(job_id)
                raise GenerationFailed(
                    f"job {job_id} exceeded {timeout}s timeout"
                )
            time.sleep(poll_interval)


# Module-level singleton — pipelines and UI import this directly.
worker = WorkerClient()


# `Tier` is re-exported here just so callers that previously did
# `from services.X.service import XService` and read `.tier` on a backend
# can still satisfy type hints when they migrate to the worker client.
__all__ = ["WorkerClient", "worker", "Tier"]
