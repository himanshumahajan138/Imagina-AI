"""Replicate-hosted image backend (FLUX.1-dev / SD 3.5 / …)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import ASPECT_RATIOS
from core.errors import GenerationFailed
from core.logger import logger
from core.replicate_client import download, first_url, run
from core.types import MediaAsset, Tier
from services.llm.prompts import IMAGE_PROMPT


def _aspect_ratio_for(dimension: str) -> str:
    """Map our internal dimension key to a FLUX-style aspect ratio."""
    a = ASPECT_RATIOS.get(dimension, "1:1")
    # FLUX accepts "16:9", "9:16", "1:1" etc. directly
    return a


class ReplicateImageBackend:
    name = "replicate"
    tier = Tier.CLOUD_OSS

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.replicate_id = cfg.get("replicate_id")
        if not self.replicate_id:
            raise GenerationFailed("Replicate image cfg missing `replicate_id`")

    def generate_image(
        self,
        prompt: str,
        out_path: Path,
        dimension: str,
        reference_images: list[Path] | None = None,
        **kwargs: Any,
    ) -> MediaAsset:
        # Build the cinematic prompt the same way the OpenAI/Gemini paths do.
        full_prompt = IMAGE_PROMPT.format(
            scene=prompt,
            script=kwargs.get("script", ""),
            video_scene=kwargs.get("video_scene", ""),
        )
        if kwargs.get("reference_text"):
            full_prompt += (
                "\n** IMPORTANT CUSTOM INSTRUCTIONS TO BE FOLLOWED FOR IMAGE GENERATION "
                f"{kwargs['reference_text']}"
            )

        logger.info(f"[replicate-image] {self.replicate_id}")
        output = run(
            self.replicate_id,
            input={
                "prompt": full_prompt,
                "aspect_ratio": _aspect_ratio_for(dimension),
                "output_format": "png",
                "output_quality": 90,
                "num_outputs": 1,
            },
        )

        url = first_url(output)
        download(url, out_path)
        return MediaAsset(path=out_path, kind="image")


def build_backend(cfg: dict[str, Any]) -> ReplicateImageBackend:
    return ReplicateImageBackend(cfg)
