"""Shared dataclasses passed between services and pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Tier(str, Enum):
    LOCAL = "local"
    CLOUD_OSS = "cloud_oss"
    API = "api"


@dataclass
class ScriptBlock:
    """One beat of a generated script."""

    script: str
    scene: str
    video_scene: str
    start_time: str
    end_time: str


@dataclass
class Script:
    blocks: list[ScriptBlock]
    theme: str | None = None
    duration: int | None = None
    language: str | None = None


@dataclass
class MediaAsset:
    path: Path
    kind: str  # "image" | "audio" | "video"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    success: bool
    asset: MediaAsset | None = None
    error: str | None = None
    backend: str | None = None
    tier: Tier | None = None
    latency_ms: int | None = None
