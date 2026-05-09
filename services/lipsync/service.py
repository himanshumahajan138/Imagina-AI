"""LipsyncService — facade over lip-sync backends."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from core.protocols import LipsyncBackend
from core.registry import pick_model, session_preferred
from core.types import MediaAsset

_BACKEND_MODULE = "services.lipsync.backends"


class LipsyncService:
    def __init__(self, model_id: str | None = None) -> None:
        preferred = model_id or session_preferred("lipsync")
        self.model_id, self.cfg = pick_model("lipsync", preferred=preferred)
        self._backend: LipsyncBackend = self._load_backend()

    def _load_backend(self) -> LipsyncBackend:
        mod = importlib.import_module(f"{_BACKEND_MODULE}.{self.cfg['backend']}")
        return mod.build_backend(self.cfg)

    def apply(
        self,
        video_path: Path,
        audio_path: Path,
        out_path: Path,
        **kwargs: Any,
    ) -> MediaAsset:
        return self._backend.apply(
            video_path=video_path, audio_path=audio_path, out_path=out_path, **kwargs
        )
