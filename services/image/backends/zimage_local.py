"""Local Z-Image-Turbo backend (Tongyi-MAI; diffusers `ZImagePipeline`).

6B-param S3-DiT model distilled to 8-step inference. Designed by Tongyi to
fit in 16 GB VRAM consumer devices, which makes it the strongest
local-tier option for an M2 16 GB box. Photorealistic, bilingual
(English + Chinese) text rendering, Apache-2.0.

Notes vs the SDXL-Turbo backend (services/image/backends/coreml_local.py):
  - `ZImagePipeline` is its own pipeline class — does NOT route through
    `AutoPipelineForText2Image`, so we import it explicitly.
  - Native bf16, `guidance_scale=0.0`, `num_inference_steps=9` (Tongyi's
    own example — the 9 results in 8 DiT forward passes).
  - Supports multiple aspect ratios out of the box.

Setup (one-time):
    pip install -U torch diffusers transformers accelerate safetensors

Diffusers must be recent enough to ship `ZImagePipeline` (added via PRs
#12703 + #12715, merged into mainline diffusers).

Weights download lazily on first call (~12 GB to ~/.cache/huggingface).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import BackendUnavailable, GenerationFailed
from core.logger import logger
from core.model_manager import get_manager
from core.types import MediaAsset, Tier


_DEFAULT_MODEL = "Tongyi-MAI/Z-Image-Turbo"


def _parse_dimension(dimension: str, fallback: tuple[int, int] = (1024, 1024)) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in dimension.lower().split("x"))
    except (ValueError, AttributeError):
        w, h = fallback
    return w, h


def _load_pipeline(model_id: str, cpu_offload: bool):
    """One-time load of the Z-Image diffusion transformer.

    Cached via `core.model_manager` so the worker keeps it resident
    across requests (~12 GB) and Streamlit reruns don't evict it.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as e:
        raise BackendUnavailable(
            "Z-Image needs torch. Run "
            "`pip install -U torch diffusers transformers accelerate safetensors`."
        ) from e

    try:
        from diffusers import ZImagePipeline  # type: ignore[import-not-found]
    except ImportError as e:
        raise BackendUnavailable(
            "ZImagePipeline not found in your diffusers install. Z-Image "
            "support landed in diffusers PRs #12703 + #12715 — upgrade with "
            "`pip install -U diffusers` (or install from main if your release "
            "predates them)."
        ) from e

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32
        logger.warning("[zimage-local] no MPS/CUDA — CPU path is impractically slow")

    logger.info(f"[zimage-local] loading {model_id} on {device} ({dtype})")
    pipe = ZImagePipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    )

    if cpu_offload and device != "cpu" and hasattr(pipe, "enable_model_cpu_offload"):
        # Offload sub-modules to CPU when not actively in the forward pass.
        # Trades latency for ~30% lower peak GPU memory — useful on 16 GB.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    logger.info(f"[zimage-local] loaded {model_id}")
    return pipe


class ZImageLocalBackend:
    """6B Z-Image-Turbo via diffusers."""

    name = "zimage_local"
    tier = Tier.LOCAL

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model_id = cfg.get("hf_id", _DEFAULT_MODEL)
        self.ram_gb = float(cfg.get("ram_gb", 12))
        self.num_inference_steps = int(cfg.get("num_inference_steps", 9))
        self.cpu_offload = bool(cfg.get("cpu_offload", False))

    def _pipe(self):
        # Shares the `diffusers::` cache prefix with SDXL-Turbo: at most
        # one image-tier model is resident at a time anyway, and
        # `evict_modality("image")` correctly drops either.
        return get_manager().get(
            f"diffusers::{self.model_id}",
            loader=lambda: _load_pipeline(self.model_id, self.cpu_offload),
            cost_gb=self.ram_gb,
        )

    def warmup(self) -> None:
        """Trigger weight load now (used by worker startup preload)."""
        self._pipe()

    def generate_image(
        self,
        prompt: str,
        out_path: Path,
        dimension: str,
        reference_images: list[Path] | None = None,
        **kwargs: Any,
    ) -> MediaAsset:
        # Z-Image-Turbo is text-only at this checkpoint. Reference-image
        # conditioning will land with `Z-Image-Edit`; ignore for now.
        pipe = self._pipe()
        width, height = _parse_dimension(dimension)

        logger.info(
            f"[zimage-local] generating {width}x{height} | prompt={prompt[:60]!r}"
        )
        try:
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=self.num_inference_steps,
                # Turbo distillation drops CFG entirely — passing >0 here
                # degrades output rather than improving it.
                guidance_scale=0.0,
            )
        except Exception as e:
            raise GenerationFailed(f"Z-Image local generation failed: {e}") from e

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(out_path)
        logger.info(f"[zimage-local] wrote {out_path}")
        return MediaAsset(
            path=out_path,
            kind="image",
            meta={"model": self.model_id, "steps": self.num_inference_steps},
        )


def build_backend(cfg: dict[str, Any]) -> ZImageLocalBackend:
    return ZImageLocalBackend(cfg)
