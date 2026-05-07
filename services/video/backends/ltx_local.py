"""Local LTX-Video 2B backend (diffusers + MPS).

Implements the `VideoBackend` protocol against `Lightricks/LTX-Video` via
Hugging Face `diffusers`. Image-to-video only — the cinematic pipeline
always passes a seed image, and LTX's text-to-video path produces less
controllable output.

Memory profile on M2 16 GB:
    bf16 weights ≈ 5 GB | inference activations ≈ 6-8 GB → ~12 GB peak.
    Tight but workable. Expect ~3-6 minutes per 8-second clip.

Setup (one-time):
    pip install torch diffusers transformers accelerate safetensors imageio imageio-ffmpeg

Weights download lazily on first call (~10 GB to ~/.cache/huggingface).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import BackendUnavailable, GenerationFailed
from core.logger import logger
from core.model_manager import get_manager
from core.types import MediaAsset, Tier


_DEFAULT_MODEL = "Lightricks/LTX-Video"
_DEFAULT_FPS = 24


def _parse_dimension(dimension: str, fallback: tuple[int, int] = (768, 768)) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in dimension.lower().split("x"))
    except (ValueError, AttributeError):
        w, h = fallback
    # LTX requires width and height divisible by 32.
    return (w // 32) * 32, (h // 32) * 32


def _round_frames(num_frames: int) -> int:
    """LTX requires (num_frames - 1) % 8 == 0 and a sensible upper bound."""
    n = max(9, min(257, int(num_frames)))
    return ((n - 1) // 8) * 8 + 1


def _load_pipeline(model_id: str):
    try:
        import torch  # type: ignore[import-not-found]
        from diffusers import LTXImageToVideoPipeline  # type: ignore[import-not-found]
    except ImportError as e:
        raise BackendUnavailable(
            "Local video backend needs torch + diffusers. Run "
            "`pip install torch diffusers transformers accelerate safetensors imageio imageio-ffmpeg`."
        ) from e

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32
        logger.warning("[ltx-local] no MPS/CUDA — CPU path is impractically slow")

    logger.info(f"[ltx-local] loading {model_id} on {device} ({dtype})")
    pipe = LTXImageToVideoPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    )
    pipe.to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    logger.info(f"[ltx-local] loaded {model_id}")
    return pipe


class LTXLocalVideoBackend:
    name = "ltx_local"
    tier = Tier.LOCAL
    produces_audio = False

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model_id = cfg.get("hf_id", _DEFAULT_MODEL)
        self.ram_gb = float(cfg.get("ram_gb", 12))
        self.fps = int(cfg.get("fps", _DEFAULT_FPS))
        self.num_inference_steps = int(cfg.get("num_inference_steps", 30))

    def _pipe(self):
        # `ltx::` prefix lets `ModelManager.evict_modality("video")` target
        # only this backend and not collide with diffusers image keys.
        return get_manager().get(
            f"ltx::{self.model_id}",
            loader=lambda: _load_pipeline(self.model_id),
            cost_gb=self.ram_gb,
        )

    def warmup(self) -> None:
        """Trigger weight load now (used by worker startup preload)."""
        self._pipe()

    def generate_video(
        self,
        prompt: str,
        out_path: Path,
        dimension: str,
        duration: float,
        seed_image: Path | None = None,
        **kwargs: Any,
    ) -> MediaAsset:
        if seed_image is None or not Path(seed_image).exists():
            raise GenerationFailed("LTX local backend requires a seed image")

        try:
            from diffusers.utils import export_to_video  # type: ignore[import-not-found]
            from PIL import Image  # already a hard dep of the project
        except ImportError as e:
            raise BackendUnavailable(f"LTX dependency missing: {e}") from e

        pipe = self._pipe()
        width, height = _parse_dimension(dimension)
        num_frames = _round_frames(int(duration * self.fps))

        logger.info(
            f"[ltx-local] generating {width}x{height} | frames={num_frames} "
            f"| fps={self.fps} | prompt={prompt[:60]!r}"
        )
        image = Image.open(seed_image).convert("RGB").resize((width, height))

        try:
            result = pipe(
                image=image,
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=self.num_inference_steps,
            )
        except Exception as e:
            raise GenerationFailed(f"LTX local video generation failed: {e}") from e

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # `result.frames` is List[List[PIL.Image]] — first batch = scene 0.
        frames = result.frames[0] if hasattr(result, "frames") else result.videos[0]
        export_to_video(frames, str(out_path), fps=self.fps)

        logger.info(f"[ltx-local] wrote {out_path}")
        return MediaAsset(
            path=out_path,
            kind="video",
            meta={
                "model": self.model_id,
                "fps": self.fps,
                "frames": num_frames,
            },
        )


def build_backend(cfg: dict[str, Any]) -> LTXLocalVideoBackend:
    return LTXLocalVideoBackend(cfg)
