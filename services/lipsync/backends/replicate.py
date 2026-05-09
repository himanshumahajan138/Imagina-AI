"""Replicate-hosted lip-sync backend (LatentSync 1.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import GenerationFailed
from core.logger import logger
from core.replicate_client import download, first_url, run
from core.types import MediaAsset, Tier


class ReplicateLipsyncBackend:
    name = "replicate"
    tier = Tier.CLOUD_OSS

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.replicate_id = cfg.get("replicate_id")
        if not self.replicate_id:
            raise GenerationFailed("Replicate lipsync cfg missing `replicate_id`")

    def apply(
        self,
        video_path: Path,
        audio_path: Path,
        out_path: Path,
        **kwargs: Any,
    ) -> MediaAsset:
        logger.info(f"[replicate-lipsync] {self.replicate_id}")

        with open(video_path, "rb") as v, open(audio_path, "rb") as a:
            output = run(
                self.replicate_id,
                input={"video": v, "audio": a},
            )

        url = first_url(output)
        download(url, out_path)
        return MediaAsset(path=out_path, kind="video")


def build_backend(cfg: dict[str, Any]) -> ReplicateLipsyncBackend:
    return ReplicateLipsyncBackend(cfg)
