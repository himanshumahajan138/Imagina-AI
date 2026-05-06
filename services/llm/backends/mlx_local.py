"""MLX-served local LLM backend (Qwen 2.5 7B Q4 by default).

Uses Apple's `mlx-lm` framework — installs cleanly on M-series Macs and
runs the prompt on the unified-memory GPU. Cached via `core.model_manager`
so a tab switch doesn't reload the weights.

Honest M2 16 GB note: Qwen 2.5 7B Q4 leaves ~10 GB free for everything
else (Streamlit + Python + libs). If you switch to a 14B class model
expect SDXL/image generation in the same session to OOM — load-on-demand
(via `model_manager`) is what saves you.
"""

from __future__ import annotations

from typing import Any

import json_repair

from core.errors import BackendUnavailable, GenerationFailed
from core.logger import logger
from core.model_manager import get_manager
from core.types import Script, Tier
from services.llm.parser import row_to_block
from services.llm.prompts import SCRIPT_PROMPT


def _seconds_for(model_type: str) -> int:
    return 8 if model_type == "gemini" else 12 if model_type == "openai" else 10


def _load_mlx(hf_id: str) -> tuple[Any, Any]:
    try:
        from mlx_lm import load  # type: ignore[import-not-found]
    except ImportError as e:
        raise BackendUnavailable(
            "mlx-lm not installed. On Apple Silicon: `pip install mlx-lm`. "
            "On non-Apple hardware this backend is unsupported."
        ) from e
    logger.info(f"[mlx-llm] loading {hf_id} (one-time, ~30 s)…")
    model, tokenizer = load(hf_id)
    logger.info(f"[mlx-llm] loaded {hf_id}")
    return model, tokenizer


class MLXLocalLLMBackend:
    name = "mlx_local"
    tier = Tier.LOCAL

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.hf_id = cfg.get("hf_id", "mlx-community/Qwen2.5-7B-Instruct-4bit")
        self.ram_gb = float(cfg.get("ram_gb", 6))

    def _model_and_tokenizer(self):
        return get_manager().get(
            f"mlx::{self.hf_id}",
            loader=lambda: _load_mlx(self.hf_id),
            cost_gb=self.ram_gb,
        )

    def generate_script(self, theme: str, duration: int, language: str, **kwargs: Any) -> Script:
        try:
            from mlx_lm import generate  # type: ignore[import-not-found]
        except ImportError as e:
            raise BackendUnavailable("mlx-lm not installed") from e

        seconds = _seconds_for(kwargs.get("model_type", "openai"))
        prompt = SCRIPT_PROMPT.format(
            theme=theme, language=language, duration=duration, seconds=seconds
        )

        # Qwen 2.5 / Llama 3 use chat templates; apply via tokenizer.
        model, tokenizer = self._model_and_tokenizer()
        messages = [
            {
                "role": "system",
                "content": "You output strict JSON only. No prose, no markdown.",
            },
            {"role": "user", "content": prompt},
        ]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        logger.info(f"[mlx-llm] generating | duration={duration}s")
        raw = generate(
            model,
            tokenizer,
            prompt=templated,
            max_tokens=4096,
            verbose=False,
        )

        try:
            structured = json_repair.loads(raw.strip())
            assert isinstance(structured, list), "expected JSON array"
            for entry in structured:
                assert "script" in entry and "scene" in entry
        except Exception as e:
            raise GenerationFailed(f"MLX LLM returned invalid script JSON: {e}") from e

        return Script(
            blocks=[row_to_block(r) for r in structured],
            theme=theme,
            duration=duration,
            language=language,
        )


def build_backend(cfg: dict[str, Any]) -> MLXLocalLLMBackend:
    return MLXLocalLLMBackend(cfg)
