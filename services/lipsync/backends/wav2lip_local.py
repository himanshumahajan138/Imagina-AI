"""Local Wav2Lip ONNX backend.

Lightweight (~500 MB checkpoint), fast on M2 with the Core ML execution
provider. Implementation outline is wired; the actual ONNX inference is
left as a TODO so the first contributor implementing this can pick a
specific Wav2Lip-ONNX repo (there are several with subtly different I/O
shapes) without me committing to one blindly.

Setup expected (Phase 4 follow-up):
    pip install onnxruntime-coreml
    python -m scripts.download_models --modality lipsync   # places model
    # under ./models/wav2lip.onnx
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import BackendUnavailable, GenerationFailed
from core.logger import logger
from core.types import MediaAsset, Tier


_DEFAULT_CKPT_PATH = Path("models") / "wav2lip.onnx"


class Wav2LipLocalBackend:
    name = "wav2lip_local"
    tier = Tier.LOCAL

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.ckpt_path = Path(cfg.get("ckpt_path", _DEFAULT_CKPT_PATH))

    def _ensure_runtime(self):
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as e:
            raise BackendUnavailable(
                "onnxruntime not installed. On macOS: "
                "`pip install onnxruntime-coreml`."
            ) from e
        if not self.ckpt_path.exists():
            raise BackendUnavailable(
                f"Wav2Lip checkpoint not found at {self.ckpt_path}. "
                "Run `python -m scripts.download_models --modality lipsync`."
            )
        return ort

    def apply(
        self,
        video_path: Path,
        audio_path: Path,
        out_path: Path,
        **kwargs: Any,
    ) -> MediaAsset:
        ort = self._ensure_runtime()
        logger.info(
            f"[wav2lip-local] {self.ckpt_path} | video={video_path} | audio={audio_path}"
        )

        # TODO: full ONNX inference path. Outline:
        #   1. ort.InferenceSession(self.ckpt_path, providers=["CoreMLExecutionProvider"])
        #   2. extract face crops + mel spectrogram chunks
        #   3. run inference per chunk, blend frames back in
        #   4. mux with original audio via ffmpeg → out_path
        #
        # Until the inference path lands, fail loudly so the registry falls
        # back to a tier-2 (Replicate) or tier-3 (Sync.so) backend.
        raise GenerationFailed(
            "Wav2Lip local inference not yet implemented. "
            "Use the Replicate (LatentSync) or Sync.so backend in the meantime."
        )


def build_backend(cfg: dict[str, Any]) -> Wav2LipLocalBackend:
    return Wav2LipLocalBackend(cfg)
