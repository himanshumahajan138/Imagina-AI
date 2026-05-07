"""VideoService — facade over video-generation backends."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from core.protocols import VideoBackend
from core.registry import pick_model, session_preferred
from core.types import MediaAsset

_BACKEND_MODULE = "services.video.backends"


class VideoService:
    def __init__(self, model_id: str | None = None) -> None:
        preferred = model_id or session_preferred("video")
        self.model_id, self.cfg = pick_model("video", preferred=preferred)
        self._backend: VideoBackend = self._load_backend()

    def _load_backend(self) -> VideoBackend:
        mod = importlib.import_module(f"{_BACKEND_MODULE}.{self.cfg['backend']}")
        return mod.build_backend(self.cfg)

    def generate_video(
        self,
        prompt: str,
        out_path: Path,
        dimension: str,
        duration: float,
        seed_image: Path | None = None,
        **kwargs: Any,
    ) -> MediaAsset:
        return self._backend.generate_video(
            prompt=prompt,
            out_path=out_path,
            dimension=dimension,
            duration=duration,
            seed_image=seed_image,
            **kwargs,
        )

    @property
    def produces_audio(self) -> bool:
        """Some backends (VEO 3) produce audio-synced video natively.

        Pipelines use this to decide whether to invoke lip-sync afterwards.
        Falls back to a `produces_audio` attribute on the backend class.
        """
        if "produces_audio" in self.cfg:
            return bool(self.cfg["produces_audio"])
        return bool(getattr(self._backend, "produces_audio", False))

    @property
    def scene_duration(self) -> int:
        """Per-scene clip length, in seconds.

        Different video backends emit different chunk sizes (Sora ≈ 12 s,
        VEO 3 ≈ 8 s, OSS ≈ 10 s). Sourced from cfg, falls back to 10.
        """
        return int(self.cfg.get("scene_duration", 10))

    @property
    def max_total_duration(self) -> int:
        """Cap on total compiled-video length for the slider UI."""
        return int(self.cfg.get("max_total_duration", self.scene_duration * 15))

    @property
    def default_total_duration(self) -> int:
        """Sensible starting value for the duration slider (= 2 scenes)."""
        return int(self.cfg.get("default_total_duration", self.scene_duration * 2))


def video_constraints() -> dict[str, int]:
    """Best-effort {scene, min, max, default, step} for the sidebar slider.

    Falls back to safe defaults when no video backend is configured (so
    the UI still renders even with empty env vars).
    """
    try:
        svc = VideoService()
    except Exception:
        return {"scene": 10, "min": 10, "max": 150, "default": 20, "step": 10}
    return {
        "scene": svc.scene_duration,
        "min": svc.scene_duration,
        "max": svc.max_total_duration,
        "default": svc.default_total_duration,
        "step": svc.scene_duration,
    }
