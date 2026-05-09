"""Local image backend — Stable Diffusion XL Turbo via diffusers + MPS.

Despite the filename ("coreml_local"), the current implementation uses
Hugging Face `diffusers` with PyTorch's MPS backend. It runs on M2
unified memory (~7 GB peak for SDXL-Turbo fp16) and produces a 1024×1024
image in roughly 6–12 s per scene.

Apple's compiled `ml-stable-diffusion` Core ML pipeline would be ~2-3×
faster on the ANE but requires a separate model conversion step and a
heavier dependency footprint. Filename retained so the yaml registry
keeps pointing at the same module — file rename can come with that
follow-up.

Setup (one-time):
    pip install torch diffusers transformers accelerate safetensors

Weights download lazily on first call (~7 GB to ~/.cache/huggingface).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import BackendUnavailable, GenerationFailed
from core.logger import logger
from core.model_manager import get_manager
from core.types import MediaAsset, Tier


_DEFAULT_MODEL = "stabilityai/sdxl-turbo"


def _load_pipeline(model_id: str, vae_id: str | None = None):
    """One-time load of an SDXL diffusers pipeline.

    Cached for the lifetime of the process via `core.model_manager`,
    so `_pipe()` returns the same object across calls and Streamlit
    reruns talking to the worker.

    `vae_id` is an optional override for the VAE — `madebyollin/sdxl-vae-fp16-fix`
    is a popular swap that produces sharper output at fp16 inference
    than the bundled SDXL VAE.
    """
    try:
        import torch  # type: ignore[import-not-found]
        from diffusers import AutoPipelineForText2Image  # type: ignore[import-not-found]
    except ImportError as e:
        raise BackendUnavailable(
            "Local image backend needs torch + diffusers. Run "
            "`pip install torch diffusers transformers accelerate safetensors`."
        ) from e

    if torch.backends.mps.is_available():
        device, dtype, variant = "mps", torch.float16, "fp16"
    elif torch.cuda.is_available():
        device, dtype, variant = "cuda", torch.float16, "fp16"
    else:
        # CPU fallback — works but is painfully slow on SDXL. Mostly here
        # so the unit-test path doesn't crash on CI runners without GPUs.
        device, dtype, variant = "cpu", torch.float32, None
        logger.warning("[diffusers-local] no MPS/CUDA — falling back to CPU (slow)")

    logger.info(f"[diffusers-local] loading {model_id} on {device} ({dtype})")
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "use_safetensors": True}
    if variant:
        kwargs["variant"] = variant
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)

    if vae_id:
        try:
            from diffusers import AutoencoderKL  # type: ignore[import-not-found]
        except ImportError as e:
            raise BackendUnavailable(f"VAE swap requested but diffusers missing: {e}") from e
        logger.info(f"[diffusers-local] swapping VAE → {vae_id}")
        pipe.vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)

    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    logger.info(f"[diffusers-local] loaded {model_id}")
    return pipe


def _parse_dimension(dimension: str, fallback: tuple[int, int] = (1024, 1024)) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in dimension.lower().split("x"))
        return w, h
    except (ValueError, AttributeError):
        return fallback


class CoreMLImageBackend:
    """SDXL local backend (diffusers+MPS — class name kept for back-compat)."""

    name = "coreml_local"
    tier = Tier.LOCAL

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model_id = cfg.get("hf_id", _DEFAULT_MODEL)
        self.ram_gb = float(cfg.get("ram_gb", 8))
        self.is_turbo = "turbo" in self.model_id.lower()
        # Quality knobs — overridable from yaml. Defaults to 4 inference
        # steps for Turbo (noticeably more detail than the official 1-step,
        # ~25s/image on M2) and 25 for full SDXL. guidance_scale=0.0 keeps
        # Turbo on its distillation rails — bumping it tends to oversharpen.
        self.num_inference_steps = int(
            cfg.get("num_inference_steps", 4 if self.is_turbo else 25)
        )
        self.guidance_scale = float(
            cfg.get("guidance_scale", 0.0 if self.is_turbo else 7.5)
        )
        self.vae_id: str | None = cfg.get("vae_id")

    def _pipe(self):
        return get_manager().get(
            # Cache key includes vae_id so swapping VAEs doesn't reuse a
            # stale pipeline.
            f"diffusers::{self.model_id}{'|vae='+self.vae_id if self.vae_id else ''}",
            loader=lambda: _load_pipeline(self.model_id, self.vae_id),
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
        # Reference-image conditioning is unsupported on this backend
        # (would need IP-Adapter weights). Silently ignored — caller still
        # gets a valid image, just without continuity hints.
        pipe = self._pipe()
        width, height = _parse_dimension(dimension)

        logger.info(
            f"[diffusers-local] generating {width}x{height} | "
            f"steps={self.num_inference_steps} guidance={self.guidance_scale} | "
            f"prompt={prompt[:60]!r}"
        )
        try:
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            )
        except Exception as e:
            raise GenerationFailed(f"Diffusers local image generation failed: {e}") from e

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(out_path)
        logger.info(f"[diffusers-local] wrote {out_path}")
        return MediaAsset(
            path=out_path,
            kind="image",
            # str() the device — `torch.device` objects aren't JSON-serializable
            # and the worker round-trips this dict through /jobs/{id}.
            meta={
                "model": self.model_id,
                "device": str(getattr(pipe, "device", "?")),
                "steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "vae": self.vae_id or "default",
            },
        )


def build_backend(cfg: dict[str, Any]) -> CoreMLImageBackend:
    return CoreMLImageBackend(cfg)
