"""LLMService — facade over LLM backends.

UI and pipelines import THIS, never a specific backend.
"""

from __future__ import annotations

import importlib
from typing import Any

from core.protocols import LLMBackend
from core.registry import pick_model, session_preferred
from core.types import Script

_BACKEND_MODULE = "services.llm.backends"


class LLMService:
    def __init__(self, model_id: str | None = None) -> None:
        preferred = model_id or session_preferred("llm")
        self.model_id, self.cfg = pick_model("llm", preferred=preferred)
        self._backend: LLMBackend = self._load_backend()

    def _load_backend(self) -> LLMBackend:
        mod_name = f"{_BACKEND_MODULE}.{self.cfg['backend']}"
        mod = importlib.import_module(mod_name)
        return mod.build_backend(self.cfg)

    def generate_script(
        self,
        theme: str,
        duration: int,
        language: str,
        **kwargs: Any,
    ) -> Script:
        return self._backend.generate_script(
            theme=theme, duration=duration, language=language, **kwargs
        )
