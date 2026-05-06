"""Backend protocols — one per modality.

Backends are *adapters*: they only know how to talk to a single provider.
No prompt construction, no parsing, no orchestration. Those live in the
service-level facade (`services/<modality>/service.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.types import MediaAsset, Script, Tier


@runtime_checkable
class LLMBackend(Protocol):
    name: str
    tier: Tier

    def generate_script(
        self,
        theme: str,
        duration: int,
        language: str,
        **kwargs: Any,
    ) -> Script: ...


@runtime_checkable
class ImageBackend(Protocol):
    name: str
    tier: Tier

    def generate_image(
        self,
        prompt: str,
        out_path: Path,
        dimension: str,
        reference_images: list[Path] | None = None,
        **kwargs: Any,
    ) -> MediaAsset: ...


@runtime_checkable
class VideoBackend(Protocol):
    name: str
    tier: Tier

    def generate_video(
        self,
        prompt: str,
        out_path: Path,
        dimension: str,
        duration: float,
        seed_image: Path | None = None,
        **kwargs: Any,
    ) -> MediaAsset: ...


@runtime_checkable
class LipsyncBackend(Protocol):
    name: str
    tier: Tier

    def apply(
        self,
        video_path: Path,
        audio_path: Path,
        out_path: Path,
        **kwargs: Any,
    ) -> MediaAsset: ...


@runtime_checkable
class TTSBackend(Protocol):
    name: str
    tier: Tier

    def synthesize(
        self,
        text: str,
        out_path: Path,
        voice: str,
        speed: float = 1.0,
        language: str = "a",
        **kwargs: Any,
    ) -> MediaAsset: ...
