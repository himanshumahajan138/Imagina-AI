"""Google Gemini backend for script generation."""

from __future__ import annotations

import os
from typing import Any

import json_repair
from google import genai
from google.genai import types as genai_types

from core.errors import GenerationFailed
from core.logger import logger
from core.types import Script, Tier
from services.llm.parser import row_to_block
from services.llm.prompts import SCRIPT_PROMPT


_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
    return _client


def _seconds_for(model_type: str) -> int:
    return 8 if model_type == "gemini" else 12 if model_type == "openai" else 10


class GeminiLLMBackend:
    name = "gemini"
    tier = Tier.API

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model = cfg.get("model", "gemini-2.5-flash")

    def generate_script(
        self, theme: str, duration: int, language: str, **kwargs: Any
    ) -> Script:
        seconds = _seconds_for(kwargs.get("model_type", "gemini"))
        prompt = SCRIPT_PROMPT.format(
            theme=theme, language=language, duration=duration, seconds=seconds
        )
        logger.info(f"[gemini-llm] {self.model} | duration={duration}s")

        try:
            response = get_client().models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    system_instruction=(
                        "You output strict JSON only. No prose, no markdown."
                    ),
                    temperature=0.7,
                ),
            )
            raw_text = (response.text or "").strip()
            structured = json_repair.loads(raw_text)
            assert isinstance(structured, list), "expected JSON array"
            for entry in structured:
                assert "script" in entry and "scene" in entry
        except Exception as e:
            raise GenerationFailed(f"Gemini script generation failed: {e}") from e

        return Script(
            blocks=[row_to_block(r) for r in structured],
            theme=theme,
            duration=duration,
            language=language,
        )


def build_backend(cfg: dict[str, Any]) -> GeminiLLMBackend:
    return GeminiLLMBackend(cfg)
