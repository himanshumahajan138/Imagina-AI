"""Local Wav2Lip ONNX backend.

End-to-end lip-sync runs entirely on the M2: ONNX Runtime with the Core ML
execution provider, OpenCV Haar cascade for face detection, librosa for
mel-spectrogram extraction, ffmpeg for the final audio mux. No remote
calls, no API keys.

Setup (one-time):
    pip install onnxruntime-coreml         # Core ML execution provider on macOS
    # or: pip install onnxruntime          # CPU fallback / non-Apple hardware
    python -m scripts.download_models --modality lipsync   # places ./models/wav2lip.onnx

Wav2Lip ONNX checkpoints exist in several variants with subtly different
I/O shapes. This implementation assumes the canonical input contract:
    face : (N, 6, 96, 96)  float32   3×RGB current frame stacked with
                                       3×RGB lower-half-masked frame
    mel  : (N, 1, 80, 16)  float32   80 mel bins × 16 time steps
    out  : (N, 3, 96, 96)  float32   lipsynced face crop, RGB
If your checkpoint differs, override `_run_session` in a subclass.
"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from core.errors import BackendUnavailable, GenerationFailed
from core.logger import logger
from core.model_manager import get_manager
from core.types import MediaAsset, Tier


_DEFAULT_CKPT_PATH = Path("models") / "wav2lip.onnx"

# Wav2Lip processing constants (canonical paper values; don't change unless
# your ONNX checkpoint was trained with different settings).
_FACE_SIZE = 96               # Wav2Lip operates on 96×96 face crops
_MEL_SR = 16000               # input sample rate for mel extraction
_MEL_N_FFT = 800
_MEL_HOP = 200
_MEL_WIN = 800
_MEL_NMELS = 80
_MEL_FMIN = 55
_MEL_FMAX = 7600
_MEL_STEP_PER_FRAME = 16      # mel time-steps consumed per video frame chunk


def _ensure_runtime():
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as e:
        raise BackendUnavailable(
            "onnxruntime not installed. macOS: `pip install onnxruntime-coreml`. "
            "Other platforms: `pip install onnxruntime`."
        ) from e
    return ort


def _load_session(ckpt_path: Path):
    ort = _ensure_runtime()
    if not ckpt_path.exists():
        raise BackendUnavailable(
            f"Wav2Lip checkpoint not found at {ckpt_path}. "
            "Run `python -m scripts.download_models --modality lipsync`."
        )
    # CoreML provider on macOS, fall back to CPU. ONNX Runtime ignores
    # providers it doesn't know about, so this list is portable.
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    logger.info(f"[wav2lip-local] loading session: {ckpt_path}")
    return ort.InferenceSession(str(ckpt_path), providers=providers)


def _compute_mel_chunks(audio_path: str, fps: float):
    """Return a list of (80 × 16) mel slices, one per video frame."""
    import librosa  # already a hard dep
    import numpy as np

    y, _sr = librosa.load(audio_path, sr=_MEL_SR)
    mel = librosa.feature.melspectrogram(
        y=y, sr=_MEL_SR, n_fft=_MEL_N_FFT, hop_length=_MEL_HOP,
        win_length=_MEL_WIN, n_mels=_MEL_NMELS,
        fmin=_MEL_FMIN, fmax=_MEL_FMAX,
    )
    log_mel = np.log(np.maximum(mel, 1e-5)).astype("float32")

    # Mel hops at _MEL_SR / _MEL_HOP frames-per-sec; align with video fps.
    mel_per_video_frame = (_MEL_SR / _MEL_HOP) / float(fps)
    chunks: list = []
    i = 0
    while True:
        start = int(i * mel_per_video_frame)
        end = start + _MEL_STEP_PER_FRAME
        if end > log_mel.shape[1]:
            # Right-align the final window so we don't run off the end.
            chunks.append(log_mel[:, log_mel.shape[1] - _MEL_STEP_PER_FRAME :])
            break
        chunks.append(log_mel[:, start:end])
        i += 1
    return chunks


def _detect_face(frame, cascade):
    """Largest face bbox in `frame` as (x, y, w, h), or None."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def _build_face_input(face_bgr):
    """Stack [original RGB, masked RGB] into a (6, 96, 96) float32 tensor in [0,1]."""
    import cv2
    import numpy as np

    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    masked = rgb.copy()
    masked[_FACE_SIZE // 2 :] = 0  # zero out the lower-half (mouth region) — Wav2Lip convention
    stacked = np.concatenate([rgb, masked], axis=2).astype("float32") / 255.0  # (96, 96, 6)
    return stacked.transpose(2, 0, 1)  # (6, 96, 96)


class Wav2LipLocalBackend:
    name = "wav2lip_local"
    tier = Tier.LOCAL

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.ckpt_path = Path(cfg.get("ckpt_path", _DEFAULT_CKPT_PATH))
        self.ram_gb = float(cfg.get("ram_gb", 1))

    def _session(self):
        return get_manager().get(
            f"wav2lip::{self.ckpt_path}",
            loader=lambda: _load_session(self.ckpt_path),
            cost_gb=self.ram_gb,
        )

    def warmup(self) -> None:
        """Trigger ONNX session load now (used by worker startup preload)."""
        self._session()

    def apply(
        self,
        video_path: Path,
        audio_path: Path,
        out_path: Path,
        **kwargs: Any,
    ) -> MediaAsset:
        try:
            import cv2
            import numpy as np
        except ImportError as e:
            raise BackendUnavailable(f"OpenCV + NumPy required: {e}") from e

        session = self._session()
        logger.info(
            f"[wav2lip-local] video={video_path} audio={audio_path} → {out_path}"
        )

        # ─── Load video frames ─────────────────────────────────────
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames: list = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise GenerationFailed(f"Could not read any frames from {video_path}")

        # ─── Audio → mel chunks aligned to video fps ───────────────
        mel_chunks = _compute_mel_chunks(str(audio_path), fps)

        # Truncate to the shorter of the two; ffmpeg mux uses the audio
        # track for final timing so a frame or two clipped is invisible.
        n = min(len(frames), len(mel_chunks))
        frames = frames[:n]
        mel_chunks = mel_chunks[:n]

        # ─── Face detection (cached cascade) ───────────────────────
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise GenerationFailed(f"Failed to load Haar cascade at {cascade_path}")

        # ─── Per-frame inference + paste-back ──────────────────────
        out_frames: list = []
        for idx, (frame, mel) in enumerate(zip(frames, mel_chunks)):
            face_box = _detect_face(frame, cascade)
            if face_box is None:
                # No face found — leave the frame untouched.
                out_frames.append(frame)
                continue

            x, y, w, h = (int(v) for v in face_box)
            face_crop = frame[y : y + h, x : x + w]
            face_resized = cv2.resize(face_crop, (_FACE_SIZE, _FACE_SIZE))

            face_input = _build_face_input(face_resized)[None]                 # (1, 6, 96, 96)
            mel_input = mel[None, None].astype("float32")                       # (1, 1, 80, 16)

            try:
                lip = session.run(None, {"face": face_input, "mel": mel_input})[0]
            except Exception as e:
                raise GenerationFailed(
                    f"Wav2Lip ONNX inference failed at frame {idx}: {e}. "
                    "Your ONNX checkpoint may use a different I/O contract."
                ) from e

            # Output is (1, 3, 96, 96) RGB float32 in [0, 1].
            lip_rgb = (lip[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
            lip_bgr = cv2.cvtColor(lip_rgb, cv2.COLOR_RGB2BGR)
            lip_resized = cv2.resize(lip_bgr, (w, h))

            out_frame = frame.copy()
            out_frame[y : y + h, x : x + w] = lip_resized
            out_frames.append(out_frame)

        # ─── Write silent video, mux audio via ffmpeg ──────────────
        height, width = out_frames[0].shape[:2]
        silent_path = Path(tempfile.gettempdir()) / f"wav2lip_silent_{uuid.uuid4()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(silent_path), fourcc, fps, (width, height))
        for f in out_frames:
            writer.write(f)
        writer.release()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(silent_path),
            "-i", str(audio_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        silent_path.unlink(missing_ok=True)
        if proc.returncode != 0:
            raise GenerationFailed(
                f"ffmpeg mux failed (rc={proc.returncode}): {proc.stderr[-500:]}"
            )

        logger.info(f"[wav2lip-local] wrote {out_path}")
        return MediaAsset(
            path=out_path,
            kind="video",
            meta={
                "ckpt": str(self.ckpt_path),
                "frames": len(out_frames),
                "fps": fps,
            },
        )


def build_backend(cfg: dict[str, Any]) -> Wav2LipLocalBackend:
    return Wav2LipLocalBackend(cfg)
