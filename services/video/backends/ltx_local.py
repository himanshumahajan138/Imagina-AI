"""Local LTX-Video 2B backend (M2-viable for short clips).

Phase 4 implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import GenerationFailed
from core.types import MediaAsset, Tier


class LTXLocalVideoBackend:
    name = "ltx_local"
    tier = Tier.LOCAL

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def generate_video(
        self,
        prompt: str,
        out_path: Path,
        dimension: str,
        duration: float,
        seed_image: Path | None = None,
        **kwargs: Any,
    ) -> MediaAsset:
        raise GenerationFailed("LTX local video backend pending Phase 4")


def build_backend(cfg: dict[str, Any]) -> LTXLocalVideoBackend:
    return LTXLocalVideoBackend(cfg)
