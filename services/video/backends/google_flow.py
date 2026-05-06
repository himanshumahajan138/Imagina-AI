"""Google Flow / VEO 3 video backend.

VEO 3 produces audio-synced video natively, so the cinematic pipeline
skips lip-sync when this backend is selected.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types

from core.config import ASPECT_RATIOS
from core.errors import GenerationFailed
from core.logger import logger
from core.types import MediaAsset, Tier


_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
    return _client


def gemini_video_generation_pipeline(
    image_path: str,
    prompt: str,
    dimension: str,
    output_path: str,
) -> str:
    """Generate a VEO 3 video given a seed image."""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    image = genai_types.Image(mime_type="image/png", image_bytes=img_bytes)

    config = genai_types.GenerateVideosConfig(
        number_of_videos=1,
        aspect_ratio=ASPECT_RATIOS[dimension],
    )
    operation = get_client().models.generate_videos(
        model="veo-3.0-generate-preview",
        prompt=prompt,
        image=image,
        config=config,
    )
    while not operation.done:
        operation = get_client().operations.get(operation)
        time.sleep(5)

    if not operation.response or not operation.response.generated_videos:
        logger.error("No video was generated.")
        raise GenerationFailed("VEO returned no video")

    video = operation.response.generated_videos[0]
    get_client().files.download(file=video.video)
    video.video.save(output_path)
    return output_path


class GoogleFlowVideoBackend:
    name = "google_flow"
    tier = Tier.API
    produces_audio = True

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
            raise GenerationFailed("VEO backend requires a seed image")
        path = gemini_video_generation_pipeline(
            image_path=str(seed_image),
            prompt=prompt,
            dimension=dimension,
            output_path=str(out_path),
        )
        return MediaAsset(path=Path(path), kind="video")


def build_backend(cfg: dict[str, Any]) -> GoogleFlowVideoBackend:
    return GoogleFlowVideoBackend(cfg)
