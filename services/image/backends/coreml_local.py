"""Local Core ML image backend (SDXL Turbo).

Phase 4 implementation. M2-friendly: uses Apple's ml-stable-diffusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import GenerationFailed
from core.types import MediaAsset, Tier


class CoreMLImageBackend:
    name = "coreml_local"
    tier = Tier.LOCAL

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def generate_image(
        self,
        prompt: str,
        out_path: Path,
        dimension: str,
        reference_images: list[Path] | None = None,
        **kwargs: Any,
    ) -> MediaAsset:
        raise GenerationFailed("Core ML image backend pending Phase 4")


def build_backend(cfg: dict[str, Any]) -> CoreMLImageBackend:
    return CoreMLImageBackend(cfg)
