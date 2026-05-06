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
