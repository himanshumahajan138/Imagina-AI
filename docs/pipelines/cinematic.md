# Cinematic Pipeline

The primary user flow. Theme → finished short film with audio, optional lipsync, watermark, and logo.

Lives at [pipelines/cinematic.py](../../pipelines/cinematic.py). ~635 lines, Streamlit-coupled.

## Phase order

Evictions happen at **phase entry** (not exit) so the previous phase's model stays warm through user iteration:

```txt
generate_script_task                (in ui/tabs/cinematic.py)
   │   loads LLM, leaves it resident
   ↓
[user iterates: edit rows, regen script, upload different SRT]
   ↓
generate_audio_images               (in pipelines/cinematic.py)
   ├── evict_models("llm")          ← phase entry: free LLM
   ├── _generate_audio
   │     ├── for each script row: _estimate_tts_speed
   │     ├──                       worker.synthesize
   │     ├──                       adjust_audio_duration(method="fit")
   │     └── concat segments → merged WAV (+ optional BGM mix)
   └── _generate_storyboard_images
         ├── for each script row: worker.generate_image
         └── (response_id chained when refinement_mode is on)
   ↓
[user iterates: regen single scene image, replace image, regen audio]
   ↓
generate_video                       (in pipelines/cinematic.py)
   ├── evict_models("image")        ← phase entry: free SDXL/Z-Image
   ├── evict_models("tts")          ← phase entry: free Kokoro
   └── _generate_single_video per scene:
         ├── worker.generate_video
         ├── if use_logo:    services.media.watermark.logo_addition
         └── if watermark:   services.media.watermark.watermark_addition
   ↓
[user iterates: regen single scene video]
   ↓
final_generation                     (in pipelines/cinematic.py)
   ├── evict_models("video")        ← phase entry: free LTX
   ├── ffmpeg concat per-scene MP4s
   ├── if lipsync_mode and not produces_audio:
   │     worker.apply_lipsync
   └── ffmpeg encode + audio mux + scale + faststart
```

Each phase is its own function; the user clicks a button to advance. Persistent state lives on `st.session_state` between phases (`script_df`, `video_data`, `scene_videos`, `final_output_path`).

## Why this is Streamlit-coupled

The functions update progress widgets (`st.progress`, `st.status`), read session state directly (`st.session_state.use_custom_audio`, `st.session_state.watermark`, etc.), and emit per-scene previews via `st.expander` / `st.image` / `st.video` mid-loop.

Splitting the pure-data orchestration from the UI rendering is on the [roadmap](../roadmap.md). For now, the trade-off is that the pipeline isn't reusable from a CLI, but writing it in Streamlit-native style means each phase emits real-time UI updates without extra plumbing.

## Worker-routed model calls

Every model call goes through `core.worker_client.worker`:

```python
from core.worker_client import worker
from core.registry import session_preferred

# script
worker.generate_script(theme=..., model_id=session_preferred("llm"))

# image (per scene)
worker.generate_image(prompt=..., model_id=session_preferred("image"))

# tts (per scene)
worker.synthesize(text=..., voice=..., model_id=session_preferred("tts"))

# video (per scene)
worker.generate_video(prompt=..., seed_image=..., model_id=session_preferred("video"))

# lipsync (final, optional)
worker.apply_lipsync(video_path=..., audio_path=..., model_id=session_preferred("lipsync"))
```

`session_preferred(modality)` reads `st.session_state.preferred_models[modality]` (set by the sidebar tier picker). If unset (user picked Auto), the worker's registry runs the auto-pick. See [registry-and-models.md](../registry-and-models.md).

## Streamlit-side post-processing

`final_generation` runs ffmpeg concat + watermark/logo overlay + final encode entirely Streamlit-side. The worker doesn't touch ffmpeg merging.

```python
# pipelines/cinematic.py:final_generation
input_streams = []
has_audio = False
for scene_info in st.session_state.scene_videos.values():
    inp = ffmpeg.input(scene_info["video_path"])
    input_streams.append(inp.video)
    if any(s["codec_type"] == "audio" for s in ffmpeg.probe(...)["streams"]):
        input_streams.append(inp.audio)
        has_audio = True

concat = ffmpeg.concat(*input_streams, v=1, a=1 if has_audio else 0).node
out = ffmpeg.output(...).overwrite_output().run(...)
```

If lipsync runs, the result is fed back through ffmpeg with `vf=final_quality` (the resolution scale string from the sidebar) for final scaling.

## Reference-image handoff

Streamlit's `UploadedFile` is in-memory bytes. The worker reads files by absolute path. Bridge: `_materialize_reference_images` persists `UploadedFile` to `/tmp` first, then passes the path:

```python
def _materialize_reference_images(refs) -> list[str]:
    out = []
    for r in refs:
        if isinstance(r, (str, Path)):
            out.append(str(r))
        elif hasattr(r, "read"):
            tmp = Path(tempfile.gettempdir()) / f"refimg_{uuid.uuid4()}.png"
            with open(tmp, "wb") as f: f.write(r.read())
            r.seek(0) if hasattr(r, "seek") else None
            out.append(str(tmp))
    return out
```

`/tmp` is shared between the two processes today (single-machine deploy). When the worker moves remote, this becomes a byte upload or a shared object store. See [roadmap.md](../roadmap.md).

## Per-scene progress UI

`generate_audio_images`:

```python
progress_bar = progress_placeholder.progress(0, text="🔄 Initializing...")
status_box = status_placeholder.status("Processing...", expanded=True)

# audio phase: 10% → 25%
audio_result = _generate_audio(script, custom_bgm)
progress_bar.progress(25, text="✅ Audio Generated")

# image phase: 30% → 90%
for i, new_img in enumerate(_generate_storyboard_images(script, dimension)):
    pct = 30 + (i + 1) * (60 // total_scenes)
    progress_bar.progress(pct, text=f"Generating Scene {i+2}...")

progress_bar.progress(100, text="✅ Audio and Images Generated Successfully.")
status_box.update(label="✅ Complete", state="complete")
```

The `_generate_storyboard_images` function is a generator yielding one scene at a time so the UI updates per scene. The worker's image gen is async (job_id polling); each iteration of the for-loop is one full POST → poll → done cycle.

## Audio sub-pipeline

See [audio-pipeline.md](../audio-pipeline.md) for the deep dive. Quick version:

1. **Pre-estimate** TTS speed from `len(text.split()) / target_seconds`.
2. **Synthesise** via `worker.synthesize(speed=tts_speed, ...)`.
3. **Force-fit** to exactly `scene_duration` via silence-pad (short) or pydub speedup (long).
4. **Concat** per-scene segments into one merged WAV.
5. **Optional BGM**: bgm_audio - 30 dB → loop to merged length → overlay merged on top.

`adjust_audio_duration(method="fit")` lives in `pipelines/cinematic.py` (not in `services/`) because it's pipeline-level logic. The actual TTS provider is in `services/tts/`.

## Phase-boundary eviction

The pipeline calls `worker.evict_models(modality=...)` at the **entry** of each phase, not the exit. This way the previous phase's model stays warm through user iteration (regenerate the script, regenerate a scene image, etc.) and only gets freed when the user has committed to moving forward.

```python
# pipelines/cinematic.py:generate_audio_images — at function entry
worker.evict_models(modality="llm")

# pipelines/cinematic.py:generate_video — at function entry
worker.evict_models(modality="image")
worker.evict_models(modality="tts")

# pipelines/cinematic.py:final_generation — at function entry
worker.evict_models(modality="video")
```

Idempotent and best-effort: HTTP failures log a warning and the pipeline continues. No-op when the upstream phase didn't load anything (e.g. user uploaded a script directly → LLM was never loaded → evict is a no-op). See [memory-management.md](../memory-management.md) for the full rationale.

## Error-recovery semantics

| Failure | Behaviour |
| --- | --- |
| Worker unreachable | `BackendUnavailable` raised by `WorkerClient`; pipeline logs and re-raises. UI shows the error. |
| Per-scene image gen fails | Logged inside the generator loop; that scene's `image_path` is left empty; pipeline continues. UI shows the error inline. |
| Per-scene video gen fails | `_generate_single_video` returns `""`; the per-scene preview shows a "❌ Video generation failed" message; pipeline continues. |
| TTS retry loop misses target | `adjust_audio_duration(method="fit")` always lands at exactly target. No silent miss. |
| Lipsync fails | `_run_lipsync` returns `None`; pipeline falls back to attaching audio without lipsync via ffmpeg `-map`. |
| ffmpeg final-encode fails | `ffmpeg.Error` caught, status banner shows "❌ Video generation failed". User can retry from "Final Merge" button. |

## Re-running and idempotence

The flow is checkpoint-driven via session state. The user can:

- Regenerate the script (replaces `script_df`).
- Edit script rows and re-run image/audio gen (replaces `editable_images`, `video_data`).
- Regenerate a single image from the storyboard gallery (modifies one entry of `editable_images`).
- Replace a generated image with a custom upload (sets `custom_image` for that row).
- Regenerate a single video from the video gallery (modifies one entry of `scene_videos`).
- Re-run final merge with different lipsync / quality settings.

The cinematic tab gates each step via session-state flags: `generating`, `action`, `video_generated`, `rerun_needed`, `image_updated`. Reading these makes the UI feel like a wizard with back-able steps.

## How a single user click drives the pipeline

A typical click sequence:

```txt
sidebar.theme = "samurai walks at night"
sidebar.duration = 24
sidebar.click "Generate Script"
   → ui/tabs/cinematic.py: st.session_state.generating = True; action = "generate_script"
   → _task_handler("generate_script", ...) runs:
       worker.generate_script(...) → DataFrame                 (LLM stays resident)
       st.session_state.script_df = ...
   → st.session_state.generating = False; rerun

(user reviews + edits the DataFrame in st.data_editor; can regenerate or upload SRT, LLM stays warm)

cinematic_tab.click "Generate Scenes and Audio"
   → action = "generate_audio_image"
   → _task_handler runs generate_audio_images(...)
       → worker.evict_models("llm")                            ← phase entry: free LLM
       → _generate_audio (with progress) → temp WAV
       → _generate_storyboard_images (with progress, per scene)
       → st.session_state.video_data = {audio_path, images}    (image+TTS stay resident)
   → rerun

(user reviews storyboard, regenerates individual images, image/TTS stay warm)

cinematic_tab.click "Generate Final Video"
   → action = "generate_final_video"
   → generate_video(video_data, duration, dimension, model_type)
       → worker.evict_models("image"), evict_models("tts")     ← phase entry: free SDXL/Z-Image + Kokoro
       → for each scene: _generate_single_video → MP4 + watermark/logo
       → st.session_state.scene_videos[N] = {...}              (video stays resident)
   → rerun

(user reviews per-scene videos, regenerates individual scenes, video stays warm)

cinematic_tab.click "Final Merge"
   → action = "merge_final"
   → final_generation(video_data, use_custom_audio, final_quality)
       → worker.evict_models("video")                          ← phase entry: free LTX
       → ffmpeg concat → merged_tmp
       → if lipsync: worker.apply_lipsync → lipsynced_path
       → ffmpeg encode + audio mux + scale → final_output
       → st.session_state.final_output_path = ...
   → rerun

(final video shown; download button available)
```

Each step is its own button + `_task_handler` invocation. The handlers set `generating=True; action=<name>` in their click handler, then the next rerun runs the actual task and resets `generating` when done. This pattern is repeated 5+ times in the cinematic tab.

## Future improvements

- **Pure-data orchestration** — split each phase into a `pure(...)` function that returns data + a `render(...)` function that does the Streamlit calls. Lets the same pipeline run from a CLI / job runner.
- **Streaming progress** — the worker's async jobs could emit progress events (`{"progress": 0.4, "message": "step 7/9"}`) that propagate up to the UI, instead of just `running` / `done`.
- **Resumable pipeline** — currently if Streamlit restarts mid-pipeline, you lose `editable_images` / `scene_videos`. Persisting these to disk (or a small SQLite store under `/tmp/imagina/`) would let the user pick up where they left off.
- **Multi-user namespacing** — session-state is per-user-tab today. If we ever serve multiple users, each pipeline run needs a unique `job_id` namespace under `/tmp` (no overlap on `scene_1.mp4`).
- **Automatic regen on script edit** — currently the user has to click "Generate Scenes" again after editing the DataFrame. We could detect changes via `hash_df` and auto-regen affected scenes.
- **Cancel button** — there's no UI to abort an in-flight pipeline phase. A "Cancel" button + worker `DELETE /jobs/{id}` would be small and high-value.
