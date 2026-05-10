"""Replicate-hosted OSS LLM backend (DeepSeek V3 / Llama 3.3 70B / …).

The `replicate_id` from `configs/models.yaml` selects which underlying
model is invoked. All of them speak the same OpenAI-style prompt input,
so the backend is uniform.
"""

from __future__ import annotations

from typing import Any

import json_repair

from core.errors import GenerationFailed
from core.logger import logger
from core.replicate_client import join_text, run
from core.types import Script, Tier
from services.llm.parser import row_to_block
from services.llm.prompts import SCRIPT_PROMPT


class ReplicateLLMBackend:
    name = "replicate"
    tier = Tier.CLOUD_OSS

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.replicate_id = cfg.get("replicate_id")
        if not self.replicate_id:
            raise GenerationFailed("Replicate LLM cfg missing `replicate_id`")

    def generate_script(self, theme: str, duration: int, language: str, **kwargs: Any) -> Script:
        seconds = int(kwargs.get("seconds") or 10)
        prompt = SCRIPT_PROMPT.format(
            theme=theme, language=language, duration=duration, seconds=seconds
        )
        logger.info(f"[replicate-llm] {self.replicate_id} | duration={duration}s")

        output = run(
            self.replicate_id,
            input={
                "prompt": prompt,
                "max_tokens": 4096,
                "temperature": 0.7,
                "system_prompt": "You output strict JSON only. No prose, no markdown.",
            },
        )
        raw_text = join_text(output).strip()

        try:
            structured = json_repair.loads(raw_text)
            assert isinstance(structured, list), "expected JSON array"
            for entry in structured:
                assert "script" in entry and "scene" in entry
        except Exception as e:
            raise GenerationFailed(f"Replicate LLM returned invalid script JSON: {e}") from e

        return Script(
            blocks=[row_to_block(r) for r in structured],
            theme=theme,
            duration=duration,
            language=language,
        )


def build_backend(cfg: dict[str, Any]) -> ReplicateLLMBackend:
    return ReplicateLLMBackend(cfg)
