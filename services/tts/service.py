"""TTSService — facade over text-to-speech backends."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from core.protocols import TTSBackend
from core.registry import pick_model, session_preferred
from core.types import MediaAsset

_BACKEND_MODULE = "services.tts.backends"


class TTSService:
    def __init__(self, model_id: str | None = None) -> None:
        preferred = model_id or session_preferred("tts")
        self.model_id, self.cfg = pick_model("tts", preferred=preferred)
        self._backend: TTSBackend = self._load_backend()

    def _load_backend(self) -> TTSBackend:
        mod = importlib.import_module(f"{_BACKEND_MODULE}.{self.cfg['backend']}")
        return mod.build_backend(self.cfg)

    def synthesize(
        self,
        text: str,
        out_path: Path,
        voice: str,
        speed: float = 1.0,
        language: str = "a",
        **kwargs: Any,
    ) -> MediaAsset:
        return self._backend.synthesize(
            text=text,
            out_path=out_path,
            voice=voice,
            speed=speed,
            language=language,
            **kwargs,
        )
