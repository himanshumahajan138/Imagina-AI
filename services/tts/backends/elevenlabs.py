"""ElevenLabs API TTS backend.

API tier alternative to Kokoro/F5. Voice IDs come from the user's
ElevenLabs library; we accept either a friendly Kokoro-style voice
(e.g. `af_heart`) and remap, or a raw ElevenLabs voice_id passed via
`voice` directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.errors import BackendUnavailable, GenerationFailed
from core.logger import logger
from core.types import MediaAsset, Tier


# Sensible defaults — ElevenLabs ships these stock voices on every account.
# Map our Kokoro-style speaker keys to a stock voice_id where reasonable;
# unknown keys are passed through unchanged so users can supply their own.
_VOICE_ID_FALLBACKS: dict[str, str] = {
    # female-leaning Kokoro voices → "Rachel"
    "af_heart": "21m00Tcm4TlvDq8ikWAM",
    "af_bella": "EXAVITQu4vr4xnSDxMaL",
    "bf_emma": "ThT5KcBeYPX3keUQqHPh",
    # male-leaning → "Adam"
    "am_adam": "pNInz6obpgDQGcFmaJgB",
    "am_michael": "VR6AewLTigWG4xSOukaG",
    "bm_george": "JBFqnCBsd6RMkjVDRZzb",
}

_DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel (universal fallback)
_DEFAULT_MODEL = "eleven_multilingual_v2"


class ElevenLabsTTSBackend:
    name = "elevenlabs"
    tier = Tier.API

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model = cfg.get("model", _DEFAULT_MODEL)

    def _client(self):
        try:
            from elevenlabs.client import ElevenLabs  # type: ignore[import-not-found]
        except ImportError as e:
            raise BackendUnavailable(
                "elevenlabs not installed. Run `pip install elevenlabs`."
            ) from e
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise BackendUnavailable("ELEVENLABS_API_KEY not set")
        return ElevenLabs(api_key=api_key)

    def _resolve_voice_id(self, voice: str) -> str:
        if not voice:
            return _DEFAULT_VOICE_ID
        # If it already looks like an ElevenLabs voice_id (long alnum), use as-is.
        if len(voice) >= 20 and voice.isalnum():
            return voice
        return _VOICE_ID_FALLBACKS.get(voice, _DEFAULT_VOICE_ID)

    def synthesize(
        self,
        text: str,
        out_path: Path,
        voice: str,
        speed: float = 1.0,
        language: str = "a",
        **kwargs: Any,
    ) -> MediaAsset:
        client = self._client()
        voice_id = self._resolve_voice_id(voice)

        logger.info(
            f"[elevenlabs-tts] voice={voice_id} model={self.model}"
        )
        try:
            stream = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id=self.model,
                text=text,
                output_format="pcm_24000",
                voice_settings={
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "speed": max(0.7, min(1.2, float(speed))),
                },
            )
            audio_bytes = b"".join(stream)
        except Exception as e:
            raise GenerationFailed(f"ElevenLabs synthesis failed: {e}") from e

        # ElevenLabs PCM is mono 16-bit @ 24 kHz; wrap in a WAV container so
        # the rest of the pipeline (which loads .wav with pydub/librosa) is
        # happy without extra muxing.
        import wave

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(audio_bytes)

        return MediaAsset(path=out_path, kind="audio")


def build_backend(cfg: dict[str, Any]) -> ElevenLabsTTSBackend:
    return ElevenLabsTTSBackend(cfg)
