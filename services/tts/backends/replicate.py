"""Replicate-hosted F5-TTS backend (zero-shot voice cloning)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import GenerationFailed
from core.logger import logger
from core.replicate_client import download, first_url, run
from core.types import MediaAsset, Tier


class ReplicateTTSBackend:
    name = "replicate"
    tier = Tier.CLOUD_OSS

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.replicate_id = cfg.get("replicate_id")
        if not self.replicate_id:
            raise GenerationFailed("Replicate TTS cfg missing `replicate_id`")

    def synthesize(
        self,
        text: str,
        out_path: Path,
        voice: str,
        speed: float = 1.0,
        language: str = "a",
        **kwargs: Any,
    ) -> MediaAsset:
        # F5-TTS expects a reference audio + reference transcript for cloning.
        # `voice` here is interpreted as a path to a reference audio file.
        ref_audio = kwargs.get("ref_audio") or voice
        ref_text = kwargs.get("ref_text", "")

        if not ref_audio or not Path(ref_audio).exists():
            raise GenerationFailed(
                "Replicate TTS (F5) needs a reference audio file in `voice` or `ref_audio`"
            )

        logger.info(f"[replicate-tts] {self.replicate_id}")
        with open(ref_audio, "rb") as f:
            output = run(
                self.replicate_id,
                input={
                    "gen_text": text,
                    "ref_audio": f,
                    "ref_text": ref_text,
                    "speed": speed,
                },
            )

        url = first_url(output)
        download(url, out_path)
        return MediaAsset(path=out_path, kind="audio")


def build_backend(cfg: dict[str, Any]) -> ReplicateTTSBackend:
    return ReplicateTTSBackend(cfg)
