"""Google Gemini / Imagen backend for cinematic scene stills."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types

from core.config import ASPECT_RATIOS
from core.errors import GenerationFailed
from core.logger import logger
from core.types import MediaAsset, Tier
from services.llm.prompts import IMAGE_PROMPT


_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
    return _client


def gemini_image_generator(item, dimension, out_path):
    """Generate a scene image via Gemini Imagen; preserved for back-compat."""
    aspect = ASPECT_RATIOS[dimension]
    custom_regenrate_text = (
        ""
        if not item.get("reference_text")
        else (
            "\n** IMPORTANT CUSTOM INSTRUCTIONS TO BE FOLLOWED FOR IMAGE GENERATION "
            f"{item.get('reference_text')}"
        )
    )
    prompt = (
        IMAGE_PROMPT.format(
            scene=item["scene"],
            script=item["script"],
            video_scene=item["video_scene"],
        )
        + custom_regenrate_text
    )

    def send_request():
        return get_client().models.generate_images(
            model="imagen-4.0-generate-preview-06-06",
            prompt=prompt,
            config=genai_types.GenerateImagesConfig(
                output_mime_type="image/png",
                aspect_ratio=aspect,
                number_of_images=1,
            ),
        )

    for attempt in range(3):
        try:
            logger.info(f"Attempt {attempt + 1} to generate image for scene")
            response = send_request()
            if response and response.generated_images:
                for generated_image in response.generated_images:
                    generated_image.image.save(out_path)
                logger.info(f"Image successfully generated at {out_path}")
                return out_path
            logger.warning(f"No images generated on attempt {attempt + 1}")
        except Exception as e:
            logger.error(f"Error generating image on attempt {attempt + 1}: {e}")
    logger.error("Failed to generate image after 3 attempts for scene")
    return None


class GeminiImageBackend:
    name = "gemini"
    tier = Tier.API

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
        item = {
            "scene": prompt,
            "script": kwargs.get("script", ""),
            "video_scene": kwargs.get("video_scene", ""),
            "reference_text": kwargs.get("reference_text"),
        }
        result = gemini_image_generator(item, dimension, str(out_path))
        if result is None:
            raise GenerationFailed("Gemini image generation returned no image")
        return MediaAsset(path=Path(result), kind="image")


def build_backend(cfg: dict[str, Any]) -> GeminiImageBackend:
    return GeminiImageBackend(cfg)
