"""ImageService — facade over image-generation backends."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from core.protocols import ImageBackend
from core.registry import pick_model, session_preferred
from core.types import MediaAsset

_BACKEND_MODULE = "services.image.backends"


class ImageService:
    def __init__(self, model_id: str | None = None) -> None:
        preferred = model_id or session_preferred("image")
        self.model_id, self.cfg = pick_model("image", preferred=preferred)
        self._backend: ImageBackend = self._load_backend()

    def _load_backend(self) -> ImageBackend:
        mod = importlib.import_module(f"{_BACKEND_MODULE}.{self.cfg['backend']}")
        return mod.build_backend(self.cfg)

    def generate_image(
        self,
        prompt: str,
        out_path: Path,
        dimension: str,
        reference_images: list[Path] | None = None,
        **kwargs: Any,
    ) -> MediaAsset:
        return self._backend.generate_image(
            prompt=prompt,
            out_path=out_path,
            dimension=dimension,
            reference_images=reference_images,
            **kwargs,
        )
