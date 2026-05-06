"""ElevenLabs API TTS backend.

Optional tier-3 alternative to Kokoro/F5. Phase 5+.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import GenerationFailed
from core.types import MediaAsset, Tier


class ElevenLabsTTSBackend:
    name = "elevenlabs"
    tier = Tier.API

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def synthesize(
        self,
        text: str,
        out_path: Path,
        voice: str,
        speed: float = 1.0,
        language: str = "a",
        **kwargs: Any,
    ) -> MediaAsset:
        raise GenerationFailed("ElevenLabs backend pending Phase 5")


def build_backend(cfg: dict[str, Any]) -> ElevenLabsTTSBackend:
    return ElevenLabsTTSBackend(cfg)
