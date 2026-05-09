# TTS Service — Voice Synthesis

Per-scene speech generation. Output is one WAV per scene which the audio pipeline force-fits to exactly `scene_duration` seconds before merging into a single track.

## Files

```
services/tts/
  __init__.py
  service.py
  backends/
    __init__.py
    kokoro_local.py    Kokoro 80M, fp32 on M2  (default local)
    replicate.py       F5-TTS via Replicate (zero-shot voice cloning)
    elevenlabs.py      ElevenLabs API
```

## Protocol method

```python
def synthesize(
    self,
    text: str,
    out_path: Path,
    voice: str,            # backend-specific id; remapped by ElevenLabs
    speed: float = 1.0,    # multiplier on natural speed
    language: str = "a",   # Kokoro lang code; ignored by API tiers
    **kwargs: Any,
) -> MediaAsset:
```

Speed pre-estimation happens **outside** the backend, in `pipelines/cinematic.py:_estimate_tts_speed`. By the time the backend gets the speed value, it's already chosen to land near `scene_duration`. See [audio-pipeline.md](../audio-pipeline.md).

## Backends

### `kokoro_local.py` — Kokoro (default local)

```yaml
kokoro:
  tier: local
  backend: kokoro_local
  ram_gb: 1
```

Lightweight (~80M params, ~1 GB resident). Loads in ~3 s. Synth is ~2-5 s per scene-line on M2.

```python
class KokoroAudioPipeline:
    def __init__(self, lang_code="a"):
        self.pipeline = KPipeline(lang_code=lang_code)
        self.sample_rate = 24000

    def text_to_audio(self, text, voice="af_heart", speed=1.0,
                      split_pattern=r"\n+", output_file=None,
                      combine_segments=True):
        generator = self.pipeline(text, voice=voice, speed=speed,
                                  split_pattern=split_pattern)
        audio_segments = [audio for _, _, audio in generator]
        combined = np.concatenate(audio_segments) if combine_segments else audio_segments
        if output_file:
            sf.write(output_file, combined, self.sample_rate)
        return combined
```

The class is the original Kokoro wrapper from before the refactor; the backend protocol is implemented by `KokoroLocalTTSBackend` below it in the same file:

```python
class KokoroLocalTTSBackend:
    name = "kokoro_local"
    tier = Tier.LOCAL

    def _pipeline(self, lang_code):
        return get_manager().get(
            f"kokoro::{lang_code}",
            loader=lambda: KokoroAudioPipeline(lang_code=lang_code),
            cost_gb=self.ram_gb,
        )

    def warmup(self, language="a"):
        self._pipeline(language)

    def synthesize(self, text, out_path, voice, speed=1.0, language="a", **kwargs):
        pipeline = self._pipeline(language)
        pipeline.text_to_audio(text, voice=voice, speed=speed, output_file=str(out_path))
        return MediaAsset(path=out_path, kind="audio")
```

**One pipeline per language**, separately cached. Switching languages mid-session means a second pipeline load. Memory cost is multiplicative (each lang ≈ 1 GB) but eviction handles it under the budget.

#### Voices

`core/config.py:SPEAKER_OPTIONS` maps friendly names to Kokoro voice ids:

```python
SPEAKER_OPTIONS = {
    "Heart": "af_heart", "Bella": "af_bella", "Nicole": "af_nicole",
    # ... 20+ voices, "a*" American female / male, "b*" British, "f*" / "i*" etc.
}
```

The script DataFrame's `speaker` column carries the friendly name; `pipelines/cinematic.py:_generate_audio` translates via `SPEAKER_OPTIONS[data["speaker"]]` before passing to `worker.synthesize`.

#### Languages

`core/config.py:COMMON_LANGUAGES`:

```python
COMMON_LANGUAGES = {
    "American English": "a", "British English": "b",
    "Spanish": "e", "French": "f", "Hindi": "h",
    "Italian": "i", "Portuguese": "p",
}
```

Each maps to a Kokoro `lang_code`. Some (Japanese 'j', Mandarin 'z') need extra deps (`pip install misaki[ja]` / `[zh]`); we don't expose them in the dropdown until those deps are documented.

### `replicate.py` — F5-TTS (zero-shot voice cloning)

```yaml
f5-tts:
  tier: cloud_oss
  backend: replicate
  replicate_id: x-lance/f5-tts
  requires_env: REPLICATE_API_TOKEN
```

Voice cloning from a reference audio sample. Different shape from Kokoro:

```python
def synthesize(self, text, out_path, voice, speed=1.0, **kwargs):
    # `voice` here is interpreted as a path to a reference audio file.
    ref_audio = kwargs.get("ref_audio") or voice
    ref_text = kwargs.get("ref_text", "")

    if not ref_audio or not Path(ref_audio).exists():
        raise GenerationFailed("Replicate TTS (F5) needs a reference audio file in `voice` or `ref_audio`")

    with open(ref_audio, "rb") as f:
        output = run(self.replicate_id, input={
            "gen_text": text, "ref_audio": f,
            "ref_text": ref_text, "speed": speed,
        })
    url = first_url(output)
    download(url, out_path)
    return MediaAsset(path=out_path, kind="audio")
```

The cinematic pipeline currently passes the Kokoro voice-id string as `voice`, which won't be a valid file path for F5. Using F5 in production requires either:

- Modifying the pipeline to pass `ref_audio=<path-to-recording>` when `model_id == "f5-tts"`, or
- Adding a per-modality "voice file" input in the sidebar that overrides the friendly-name dropdown when F5 is picked.

Today F5 is selectable but using it end-to-end needs the second customisation. Filed as future improvement.

### `elevenlabs.py` — ElevenLabs API

```yaml
elevenlabs:
  tier: api
  backend: elevenlabs
  requires_env: ELEVENLABS_API_KEY
```

Mapping table from Kokoro-style friendly voice id to ElevenLabs `voice_id` so the same picker works without UI changes:

```python
_VOICE_ID_FALLBACKS = {
    "af_heart":  "21m00Tcm4TlvDq8ikWAM",   # Rachel
    "af_bella":  "EXAVITQu4vr4xnSDxMaL",
    "bf_emma":   "ThT5KcBeYPX3keUQqHPh",
    "am_adam":   "pNInz6obpgDQGcFmaJgB",
    "am_michael":"VR6AewLTigWG4xSOukaG",
    "bm_george": "JBFqnCBsd6RMkjVDRZzb",
}
_DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel
```

Unmapped voices fall back to Rachel. If the user passes a raw ElevenLabs voice_id (≥20 alnum chars), we use it as-is.

```python
stream = client.text_to_speech.convert(
    voice_id=voice_id,
    model_id=self.model,             # eleven_multilingual_v2 default
    text=text,
    output_format="pcm_24000",       # 24 kHz, 16-bit PCM
    voice_settings={"stability": 0.5, "similarity_boost": 0.75,
                    "speed": max(0.7, min(1.2, float(speed)))},
)
audio_bytes = b"".join(stream)
```

Then wrap PCM in a WAV container manually so downstream `pydub.AudioSegment.from_wav` works:

```python
import wave
with wave.open(str(out_path), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes(audio_bytes)
```

Why not request `mp3_44100` from ElevenLabs and skip the WAV wrap? `pydub` would handle MP3 fine, but: WAV is lossless, the rest of the pipeline assumes WAV (`AudioSegment.from_wav` is hardcoded), and we already have `wave` in stdlib.

## Speed pre-estimation

The pipeline pre-estimates `tts_speed` so the synth lands close to target on the first call (no retry loop). See [audio-pipeline.md](../audio-pipeline.md).

```python
# pipelines/cinematic.py
def _estimate_tts_speed(text, target_seconds, wps=2.4):
    words = len(text.split())
    if words == 0 or target_seconds <= 0:
        return 1.0
    natural_seconds = words / wps
    return max(0.7, min(1.4, natural_seconds / target_seconds))
```

Then `worker.synthesize(speed=tts_speed)`. If the user manually edited the `speed` column in the script DataFrame to anything other than 1.0, that wins; otherwise the auto-estimate is used.

After synth, `adjust_audio_duration(method="fit")` does silence-padding (short) or pydub speedup (long) to land at exactly `scene_duration`.

## Memory & eviction

`kokoro::` cache prefix. Per-language pipeline cached separately. The cinematic pipeline calls `worker.evict_models(modality="tts")` after image gen completes to free Kokoro before video gen needs the slot. Kokoro is small (~1 GB) so this isn't critical for OOM avoidance, but it's good hygiene.

## Failure modes

| Failure | What's raised |
| --- | --- |
| `kokoro` package not installed | top-level `from kokoro import KPipeline` raises ImportError; the backend module fails to import; `BackendUnavailable("...")` from the service-layer wrapper |
| Kokoro language code unknown | underlying `KPipeline` raises; surfaces as runtime exception |
| F5-TTS missing ref audio | `GenerationFailed("Replicate TTS (F5) needs a reference audio file ...")` |
| ElevenLabs `elevenlabs` not installed | `BackendUnavailable("elevenlabs not installed. Run pip install elevenlabs.")` |
| ElevenLabs missing API key | `BackendUnavailable("ELEVENLABS_API_KEY not set")` |
| ElevenLabs API error (quota / 401) | `GenerationFailed(f"ElevenLabs synthesis failed: {e}")` |

## Failure to fit duration

If the TTS produces audio nowhere near target despite the speed estimate (e.g. a non-English script triggers a very slow Kokoro voice), the post-fit pass still lands the segment at exactly `scene_duration` via silence-pad or pydub speedup. See [audio-pipeline.md](../audio-pipeline.md).

The pipeline logs the actual vs target so you can see drift in the worker log:

```
[audio-gen] scene 1: tts_speed=0.85 target=8s actual=8.000s
```

## Future improvements

- **F5-TTS proper integration** — sidebar input for "reference voice clip" + per-scene optional `ref_audio`. Currently F5 is selectable but unusable end-to-end.
- **Streaming TTS** — chunk synthesis as the LLM emits the script, instead of one-call-per-scene. Would shave audio gen time when scripts are long.
- **Per-scene voice override** — `voice` column in the data editor already exists; the picker reads it. Could surface a per-scene voice selector with previews.
- **Prosody / emotion controls** — Kokoro and ElevenLabs both support some emotion control; not exposed via cfg.
- **Sample rate consistency** — Kokoro emits 24 kHz, ElevenLabs we wrap as 24 kHz, F5 emits whatever Replicate returns. Force a uniform 24 kHz with a downstream resample to avoid muxer surprises.
- **Drop the WAV-wrap on ElevenLabs** — request `wav` directly if the SDK exposes it; one less buffer.
- **Multi-speaker batching** — currently one synth call per scene. If we ever want multi-speaker dialogue per scene, the protocol needs `text` to become `list[(speaker, line)]` with a different signature.
- **Voice library refresh** — Kokoro adds voices over time; the static `SPEAKER_OPTIONS` dict will go stale. A yaml or registry-driven voice list would be more maintainable.
