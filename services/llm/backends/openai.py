"""OpenAI / GPT backend for script generation."""

from __future__ import annotations

import os
from typing import Any

import json_repair
from openai import OpenAI

from core.errors import GenerationFailed
from core.logger import logger
from core.types import Script, Tier
from services.llm.parser import row_to_block
from services.llm.prompts import SCRIPT_PROMPT


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def openai_script_generator(
    theme: str,
    language: str,
    duration: int,
    model: str = "gpt-5.4-mini",
    model_type: str = "openai",
) -> list[dict]:
    """Generate a list of cinematic script beats via OpenAI.

    Public function preserved for back-compat with core/utils.py callers.
    """
    seconds = 8 if model_type == "gemini" else 12 if model_type == "openai" else 10

    prompt = SCRIPT_PROMPT.format(
        theme=theme, language=language, duration=duration, seconds=seconds
    )
    try:
        response = get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        raw_output = response.choices[0].message.content.strip()
        structured = json_repair.loads(raw_output)
        assert isinstance(structured, list), "Expected a list of dictionaries"
        for entry in structured:
            assert "script" in entry and "scene" in entry
        return structured
    except Exception as e:
        logger.error("OpenAI script generation failed.")
        raise e


_row_to_block = row_to_block  # back-compat re-export


class OpenAILLMBackend:
    name = "openai"
    tier = Tier.API

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def generate_script(self, theme: str, duration: int, language: str, **kwargs: Any) -> Script:
        try:
            raw = openai_script_generator(
                theme=theme,
                language=language,
                duration=duration,
                model=self.cfg.get("model", "gpt-5.4-mini"),
                model_type=kwargs.get("model_type", "openai"),
            )
        except Exception as e:
            raise GenerationFailed(f"OpenAI script generation failed: {e}") from e

        return Script(
            blocks=[row_to_block(r) for r in raw],
            theme=theme,
            duration=duration,
            language=language,
        )


def build_backend(cfg: dict[str, Any]) -> OpenAILLMBackend:
    return OpenAILLMBackend(cfg)
