# Audio Pipeline

How spoken script lines become a single perfectly-timed audio track that aligns frame-for-frame with the per-scene video.

## The constraint

Each video scene has a fixed length determined by the active video model (`scene_duration` in yaml — 8s for VEO 3, 12s for Sora, 10s for OSS). The audio for scene N must be **exactly** `scene_duration` seconds long. Cumulative drift across N scenes would desync the final video.

Naive approach: synthesise speech, accept whatever duration it produces, hope it's close. Doesn't work — TTS speed varies by language, voice, and content. A 3-word line becomes 1.5 s; a 30-word line becomes 12 s. Both miss target.

## Two-stage approach

```txt
text + target_duration
   ↓
[1] _estimate_tts_speed(text, target)         # pre-estimate
   ↓
worker.synthesize(text, speed=estimated)      # one TTS call, lands close to target
   ↓
[2] adjust_audio_duration(method="fit")       # post-fit: short → silence pad
                                              #          long  → pydub speedup
   ↓
audio segment of EXACTLY target_duration
```

Stage 1 gets the audio close to target on the first try (no retry loop, no wasted TTS calls). Stage 2 is a hard guarantee — silence-pad if short, pitch-preserving speedup if long. Output is exactly `target_duration` to the millisecond.

## Stage 1: pre-estimate speed

```python
# pipelines/cinematic.py
_WORDS_PER_SECOND_AT_NORMAL = 2.4
_TTS_SPEED_MIN, _TTS_SPEED_MAX = 0.7, 1.4

def _estimate_tts_speed(text: str, target_seconds: float) -> float:
    words = len(text.split())
    if words == 0 or target_seconds <= 0:
        return 1.0
    natural_seconds = words / _WORDS_PER_SECOND_AT_NORMAL
    speed = natural_seconds / target_seconds
    return max(_TTS_SPEED_MIN, min(_TTS_SPEED_MAX, speed))
```

### Why 2.4 wps

English cinematic narration runs slightly slower than conversational speech (~150 WPM ≈ 2.5 wps). Picking 2.4 centres the estimate so most reasonable script lines land within the safe ±15% speed band Stage 2 can correct effortlessly. Tuned by trial; not provider-specific.

### Why clamp to [0.7, 1.4]

Beyond ~1.5× speed, TTS engines produce robotic / slurred output. Below ~0.7×, prosody falls apart. Stage 2 picks up whatever Stage 1 couldn't — extreme cases (1-word line for 30s, or 100-word paragraph for 5s) get silence-padded or speedup'd respectively, with quality consequences but no duration drift.

### User override

If the user has manually set the `speed` column in the script DataFrame to a value other than 1.0, that wins:

```python
# pipelines/cinematic.py:_generate_audio
user_speed = data.get("speed")
if user_speed and float(user_speed) != 1.0:
    tts_speed = float(user_speed)
else:
    tts_speed = _estimate_tts_speed(data["script"], duration)
```

So `speed = 1.0` (the default in the data editor) means "auto"; anything else is explicit.

## Stage 2: force-fit to exact duration

```python
# pipelines/cinematic.py:adjust_audio_duration with method="fit"
if method == "fit":
    target_ms = int(target_duration * 1000)
    current_ms = len(audio)
    diff_ms = target_ms - current_ms

    if diff_ms == 0:
        return audio_path, audio, False, 1.0

    if diff_ms > 0:
        # SHORT → silence-pad at the end
        audio = audio + AudioSegment.silent(duration=diff_ms)
    else:
        # LONG → pydub.speedup (phase-vocoder, preserves pitch)
        speed_factor = current_duration / target_duration
        audio = audio.speedup(playback_speed=speed_factor)
        # speedup's output length isn't always exact — top up or trim
        if len(audio) < target_ms:
            audio = audio + AudioSegment.silent(duration=target_ms - len(audio))
        elif len(audio) > target_ms:
            audio = audio[:target_ms]

    audio.export(str(output_path), format="wav")
    return str(output_path), audio, False, 1.0
```

### Why short → silence pad

A pause at the end of the line is the most natural-sounding thing for "we have spare time". The user's intended content was already spoken; the silence is just dead air. Alternatives we rejected:

- **Time-stretch slower** — librosa `time_stretch(rate < 1)` works but drags speech artificially.
- **Repeat the line** — uncanny.
- **Add filler ("um", "ah")** — quality hit.

### Why long → speedup, not librosa stretch

`pydub.AudioSegment.speedup(playback_speed)` uses overlap-add with a phase vocoder. Speech still sounds like the same speaker, just talking faster. `librosa.effects.time_stretch` works similarly but introduces audible artifacts on speech at >1.2× compression.

We use librosa elsewhere (`method="smart"` legacy path), but for the canonical "fit" rule the user asked for the simpler "make it fast" semantics. Pydub's speedup is the cleanest implementation of that.

### Why the trim/pad afterwards

`pydub.speedup`'s output length varies by ±20-50 ms due to its windowing math. After speedup we explicitly land on `target_ms` via final `audio + silence` or `audio[:target_ms]`. The audible content is the speedup; the trim is sub-perceptual.

### Tested behaviour

Across input lengths 1s, 5s, 6s, 7s, 7.95s, 8s, 8.05s, 9s, 10s, 12s, 14s targeting 8s:

| Input | Output | Δ from target |
| --- | --- | --- |
| 1.000 s | 8.000 s | 0.00 ms |
| 5.000 s | 8.000 s | 0.00 ms |
| 7.000 s | 8.000 s | 0.00 ms |
| 7.950 s | 8.000 s | 0.00 ms |
| 8.000 s | 8.000 s | 0.00 ms |
| 8.050 s | 8.000 s | 0.00 ms |
| 9.000 s | 8.000 s | 0.00 ms |
| 12.000 s | 8.000 s | 0.00 ms |
| 14.000 s | 8.000 s | 0.00 ms |

All sub-millisecond. Total of 11 segments at 8s target = 88.000 s exact.

## Wiring in `_generate_audio`

```python
# pipelines/cinematic.py:_generate_audio
def _generate_audio(script, custom_bgm=None):
    language = st.session_state.get("language")
    lang_code = COMMON_LANGUAGES[language]
    duration = _scene_duration()                    # from VideoService
    tts_model_id = session_preferred("tts")

    audio_segments = []
    for index, data in enumerate(script):
        # Stage 0: choose tts_speed (user override OR auto-estimate)
        user_speed = data.get("speed")
        if user_speed and float(user_speed) != 1.0:
            tts_speed = float(user_speed)
        else:
            tts_speed = _estimate_tts_speed(data["script"], duration)

        # Single TTS call
        temp_audio_path = Path(tempfile.gettempdir()) / f"generated_{uuid.uuid4()}.wav"
        worker.synthesize(
            text=data["script"], out_path=temp_audio_path,
            voice=SPEAKER_OPTIONS[data["speaker"]],
            speed=tts_speed, language=lang_code,
            model_id=tts_model_id,
        )

        # Stage 2: force-fit to exactly `duration`
        fit_path, fit_seg, _, _ = adjust_audio_duration(
            str(temp_audio_path), target_duration=duration, method="fit",
        )
        audio_segments.append(fit_seg)
        logger.info(
            f"[audio-gen] scene {index + 1}: tts_speed={tts_speed:.2f} "
            f"target={duration}s actual={len(fit_seg)/1000:.3f}s"
        )

    # Concat + optional BGM mix
    merged_audio = audio_segments[0]
    for segment in audio_segments[1:]:
        merged_audio += segment
    ...
```

Each scene logs its actual vs target so you can verify drift in the worker log.

## BGM mixing

If the user uploads custom background music, it's mixed under the merged speech track:

```python
if custom_bgm:
    if hasattr(custom_bgm, "read"):                            # Streamlit UploadedFile
        bgm_path = Path(tempfile.gettempdir()) / f"{custom_bgm.name}_{uuid.uuid4()}.wav"
        with open(bgm_path, "wb") as f: f.write(custom_bgm.read())
    else:
        bgm_path = Path(custom_bgm)                            # already a path

    bgm = AudioSegment.from_wav(bgm_path) - 30                 # attenuate by 30 dB
    if len(bgm) < len(merged_audio):
        bgm *= (len(merged_audio) // len(bgm)) + 1             # loop to cover length
    bgm = bgm[: len(merged_audio)]                             # trim to merged length
    final_audio = bgm.overlay(merged_audio)                    # speech on top of BGM
    final_audio.export(mixed_audio_path, format="wav")
```

`-30 dB` attenuation puts BGM well below the speech volume. `*=` on `AudioSegment` repeats; `bgm[:len]` trims. The merged speech sits on top via `bgm.overlay(speech)`.

## Lipsync handoff

The merged audio + concatenated video go to lipsync as separate inputs:

```python
# pipelines/cinematic.py:final_generation
if (st.session_state.lipsync_mode
    and audio_path and Path(audio_path).exists()
    and not native_audio):                                     # VEO 3 → skip
    lipsynced_video_path = _run_lipsync(merged_tmp, audio_path)
```

`_run_lipsync` calls `worker.apply_lipsync` which routes to Sync.so / LatentSync / Wav2Lip. See [services/lipsync.md](services/lipsync.md).

If the active video backend has `produces_audio: true` (currently only VEO 3), lipsync is skipped — VEO already produces the synced output and a second sync would only add artifacts.

## When NOT to use the audio pipeline

The cinematic pipeline can also run without TTS:

- `use_custom_audio = False` (sidebar toggle off): no audio generation. Final video is silent OR uses VEO's native audio.
- VEO 3 with `use_custom_audio=False`: the VEO clip's native audio carries through ffmpeg concat unchanged.

Both paths are gated in `final_generation` via `if use_custom_audio:` blocks.

## Edge cases

| Case | Behaviour |
| --- | --- |
| Empty script line | `_estimate_tts_speed` returns 1.0; TTS produces near-silence; fit pads with silence to target. |
| 1-word line for 30s scene | speed clamped to 0.7; TTS produces ~5s of audio; fit pads ~25s of silence. |
| 100-word paragraph for 5s scene | speed clamped to 1.4; TTS produces ~30s; fit speeds up 6× — robotic but on-target. |
| BGM shorter than merged audio | looped via `bgm *= n+1` then trimmed. Loop boundary is audible; user usually picks BGM at or longer than merge length. |
| BGM longer than merged | trimmed to length. |
| Kokoro language switch mid-script | not exposed in UI (single language per script). If it happened, each language would load its own pipeline (~1 GB each) and the LRU would handle eviction. |

## Failure modes

| Failure | Behaviour |
| --- | --- |
| TTS backend raises | exception bubbles; `_generate_audio` returns `{"error": str(e)}`; UI shows the message. |
| BGM file unreadable | pydub raises; caught and surfaces as "Audio generation failed". |
| `adjust_audio_duration("fit")` somehow misses target | (shouldn't happen — verified); the trim/pad fallback still lands at exact `target_ms`. |
| Worker unreachable | `BackendUnavailable`; logged; UI shows the worker-down banner. |

## Future improvements

- **Crossfade between scenes** — currently per-scene segments concat with hard cuts. A 100-200 ms crossfade per boundary would smooth over speedup'd segments.
- **Per-scene BGM mood** — single global BGM today. Could let the user pick BGM per scene from a library.
- **Loop-aware BGM** — detect natural loop points instead of hard-trim at end. Tools like `essentia` can find them.
- **Voice-aware speed estimation** — Kokoro voices have different baseline speeds. Per-voice `wps` calibration would improve Stage 1's accuracy.
- **Post-fit quality preset** — "graceful" mode could allow slight target drift (±200 ms) to avoid speedup artifacts on >1.3× compression.
- **Mute scenes** — a `mute` column in the data editor that produces silent segments without invoking TTS at all.
- **Audio normalisation** — RMS-normalise each segment before concat so BGM mix volume is consistent across scenes.
- **Stream concat to disk** — currently we hold all `AudioSegment` objects in RAM until concat. For very long scripts (50+ scenes), stream to a single growing WAV instead.
