"""OpenAI gpt-image-1 backend (cinematic scene stills)."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from core.errors import GenerationFailed
from core.logger import logger
from core.types import MediaAsset, Tier
from services.llm.prompts import IMAGE_PROMPT


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def openai_image_generator(item, dimension, out_path, previous_response_id=None):
    """Generate one cinematic image; preserved for back-compat."""
    custom_regenrate_text = (
        ""
        if not item.get("reference_text")
        else (
            "\n** IMPORTANT CUSTOM INSTRUCTIONS TO BE FOLLOWED FOR IMAGE GENERATION "
            f"{item.get('reference_text')}"
        )
    )
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": IMAGE_PROMPT.format(
                scene=item["scene"],
                script=item["script"],
                video_scene=item["video_scene"],
            )
            + custom_regenrate_text,
        }
    ]

    custom_images_list = item.get("reference_images") or []
    for img_file in custom_images_list:
        # Accept either an UploadedFile-like object or a path string.
        if hasattr(img_file, "read"):
            raw = img_file.read()
            if hasattr(img_file, "seek"):
                img_file.seek(0)
        else:
            with open(img_file, "rb") as f:
                raw = f.read()
        base64_image = base64.b64encode(raw).decode("utf-8")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base64_image}",
            }
        )

    if item.get("old_image"):
        old_image_path = item["old_image"]
        if os.path.exists(old_image_path):
            with open(old_image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{base64_image}",
                }
            )
        else:
            logger.warning(f"Old image not found at: {old_image_path}")

    response = get_client().responses.create(
        model="gpt-image-2",
        input=[{"role": "user", "content": content}],
        tools=[
            {
                "type": "image_generation",
                "background": "opaque",
                "quality": "high" if custom_images_list else "medium",
                "size": dimension,
            }
        ],
        previous_response_id=previous_response_id or None,
    )

    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]

    if image_data:
        image_base64 = image_data[0]
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(image_base64))

    return out_path, response.id


class OpenAIImageBackend:
    name = "openai"
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
            "reference_images": reference_images or [],
            "reference_text": kwargs.get("reference_text"),
            "old_image": kwargs.get("old_image"),
        }
        try:
            path, response_id = openai_image_generator(
                item,
                dimension,
                str(out_path),
                previous_response_id=kwargs.get("previous_response_id"),
            )
        except Exception as e:
            raise GenerationFailed(f"OpenAI image generation failed: {e}") from e
        return MediaAsset(
            path=Path(path),
            kind="image",
            meta={"response_id": response_id},
        )


def build_backend(cfg: dict[str, Any]) -> OpenAIImageBackend:
    return OpenAIImageBackend(cfg)
