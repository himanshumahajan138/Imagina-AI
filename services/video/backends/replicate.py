"""Replicate-hosted video backend (Wan 2.1 / HunyuanVideo / LTX / …).

Routes by the `replicate_id` in cfg. Each underlying model takes slightly
different input keys; we build the safest superset and pass it. Replicate
ignores unknown keys, so this is fine in practice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import ASPECT_RATIOS
from core.errors import GenerationFailed
from core.logger import logger
from core.replicate_client import download, first_url, run
from core.types import MediaAsset, Tier


class ReplicateVideoBackend:
    name = "replicate"
    tier = Tier.CLOUD_OSS
    produces_audio = False

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.replicate_id = cfg.get("replicate_id")
        if not self.replicate_id:
            raise GenerationFailed("Replicate video cfg missing `replicate_id`")

    def generate_video(
        self,
        prompt: str,
        out_path: Path,
        dimension: str,
        duration: float,
        seed_image: Path | None = None,
        **kwargs: Any,
    ) -> MediaAsset:
        if seed_image is None or not seed_image.exists():
            raise GenerationFailed("Replicate video backends expect a seed image")

        logger.info(
            f"[replicate-video] {self.replicate_id} | duration={duration}s "
            f"| dimension={dimension}"
        )

        with open(seed_image, "rb") as image_file:
            output = run(
                self.replicate_id,
                input={
                    "prompt": prompt,
                    "image": image_file,
                    "aspect_ratio": ASPECT_RATIOS.get(dimension, "16:9"),
                    # ~24 fps × duration; Replicate models clamp this internally.
                    "num_frames": max(16, int(duration * 24)),
                    # Wan / Hunyuan accept these; LTX ignores them.
                    "num_inference_steps": 25,
                    "guidance_scale": 5.0,
                },
            )

        url = first_url(output)
        download(url, out_path)
        return MediaAsset(path=out_path, kind="video")


def build_backend(cfg: dict[str, Any]) -> ReplicateVideoBackend:
    return ReplicateVideoBackend(cfg)
