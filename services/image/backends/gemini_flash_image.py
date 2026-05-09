"""Google Gemini 2.5 Flash Image (Nano Banana) backend.

Free-tier image generation via the Gemini multimodal `generate_content` API.
Unlike Imagen, Nano Banana takes images as additional content parts (native
multi-image conditioning) and infers aspect from a textual hint in the prompt.
"""

from __future__ import annotations

import base64
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


def _load_image_bytes(src: Any) -> bytes | None:
    if src is None:
        return None
    if hasattr(src, "read"):
        raw = src.read()
        if hasattr(src, "seek"):
            src.seek(0)
        return raw
    path = str(src)
    if not os.path.exists(path):
        logger.warning(f"[gemini-flash-image] reference not found: {path}")
        return None
    with open(path, "rb") as f:
        return f.read()


def _image_part(raw: bytes) -> genai_types.Part:
    return genai_types.Part.from_bytes(data=raw, mime_type="image/png")


class GeminiFlashImageBackend:
    name = "gemini_flash_image"
    tier = Tier.API

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model = cfg.get("model", "gemini-2.5-flash-image")

    def generate_image(
        self,
        prompt: str,
        out_path: Path,
        dimension: str,
        reference_images: list[Path] | None = None,
        **kwargs: Any,
    ) -> MediaAsset:
        aspect = ASPECT_RATIOS.get(dimension, "1:1")
        for x in range(10):
            print(aspect)
        custom = kwargs.get("reference_text")
        custom_block = (
            ""
            if not custom
            else f"\n** IMPORTANT CUSTOM INSTRUCTIONS TO BE FOLLOWED FOR IMAGE GENERATION {custom}"
        )
        prompt_text = (
            IMAGE_PROMPT.format(
                scene=prompt,
                script=kwargs.get("script", ""),
                video_scene=kwargs.get("video_scene", ""),
            )
            + custom_block
            + f"\n\nRender at aspect ratio {aspect} ({dimension})."
        )

        contents: list[Any] = [prompt_text]
        for ref in reference_images or []:
            raw = _load_image_bytes(ref)
            if raw:
                contents.append(_image_part(raw))
        old_raw = _load_image_bytes(kwargs.get("old_image"))
        if old_raw:
            contents.append(_image_part(old_raw))

        try:
            response = get_client().models.generate_content(
                model=self.model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )
        except Exception as e:
            raise GenerationFailed(f"Gemini Flash Image generation failed: {e}") from e

        image_bytes: bytes | None = None
        for cand in response.candidates or []:
            for part in (cand.content.parts or []) if cand.content else []:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    data = inline.data
                    image_bytes = data if isinstance(data, bytes) else base64.b64decode(data)
                    break
            if image_bytes:
                break

        if not image_bytes:
            raise GenerationFailed("Gemini Flash Image returned no image data")

        with open(out_path, "wb") as f:
            f.write(image_bytes)
        logger.info(f"[gemini-flash-image] saved → {out_path}")

        return MediaAsset(path=Path(out_path), kind="image", meta={"model": self.model})


def build_backend(cfg: dict[str, Any]) -> GeminiFlashImageBackend:
    return GeminiFlashImageBackend(cfg)
