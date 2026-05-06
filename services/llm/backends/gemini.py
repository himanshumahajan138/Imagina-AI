"""Google Gemini backend for script generation.

Phase 1 wires Gemini script generation here (currently absent in utils.py;
script gen today only uses OpenAI).
"""

from __future__ import annotations

from typing import Any

from core.errors import GenerationFailed
from core.types import Script, Tier


class GeminiLLMBackend:
    name = "gemini"
    tier = Tier.API

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def generate_script(self, theme: str, duration: int, language: str, **kwargs: Any) -> Script:
        raise GenerationFailed("Not yet implemented in Phase 0")


def build_backend(cfg: dict[str, Any]) -> GeminiLLMBackend:
    return GeminiLLMBackend(cfg)
