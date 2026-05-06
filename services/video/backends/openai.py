"""OpenAI Sora video backend (image-to-video)."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from core.config import SORA_DIMENSIONS
from core.errors import GenerationFailed
from core.logger import logger
from core.types import MediaAsset, Tier
from services.media.watermark import crop_image_to_dimension


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def sora_video_generation_pipeline(
    image_path: str,
    prompt: str,
    dimension: str,
    duration: int,
    output_path: str | None = None,
):
    """Generate a video via Sora given a seed image; preserved for back-compat."""
    logger.info(f"Starting video generation: {prompt[:50]}...")

    cropped_image_path = crop_image_to_dimension(image_path, dimension)

    video = get_client().videos.create(
        model="sora-2",
        prompt=prompt,
        size=dimension,
        input_reference=Path(cropped_image_path),
        seconds=str(duration),
        timeout=600,
    )

    logger.info(f"Video ID: {video.id} - Status: {video.status}")

    while video.status in ("in_progress", "queued"):
        logger.info(f"Polling status for generation {video.id}")
        video = get_client().videos.retrieve(video.id)
        logger.info(f"Video ID: {video.id} - Status: {video.status}")
        time.sleep(5)

    if video.status == "failed":
        logger.error(f"Sora Video generation failed: {video.id}; Reason: {video.error}")
        raise GenerationFailed(
            f"Sora Video generation failed: {video.id}; Reason: {video.error}"
        )

    logger.info("Sora Video generation completed")
    content = get_client().videos.download_content(video.id, variant="video")

    if not output_path:
        temp_fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="sora_")
        os.close(temp_fd)

    content.write_to_file(output_path)
    logger.info(f"Video saved: {output_path}")

    if cropped_image_path != image_path:
        try:
            os.remove(cropped_image_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary image: {e}")

    return output_path


class OpenAIVideoBackend:
    name = "openai"
    tier = Tier.API
    produces_audio = False

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
        if seed_image is None:
            raise GenerationFailed("Sora backend requires a seed image")
        # Sora uses provider-specific dim strings (1280x720 etc.) — map from canonical
        sora_dim = SORA_DIMENSIONS.get(dimension, dimension)
        path = sora_video_generation_pipeline(
            image_path=str(seed_image),
            prompt=prompt,
            dimension=sora_dim,
            duration=int(duration),
            output_path=str(out_path),
        )
        return MediaAsset(path=Path(path), kind="video")


def build_backend(cfg: dict[str, Any]) -> OpenAIVideoBackend:
    return OpenAIVideoBackend(cfg)
