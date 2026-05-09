# Pipelines

`pipelines/` is the orchestration layer between the UI and the services / worker. It's where multiple modalities are stitched together into the user-facing flows.

```
pipelines/
  cinematic.py      Theme → script → image → video → tts → merge.
  srt.py            User-supplied SRT block → list of script dicts (re-export shim)
```

## What "pipeline" means here

A pipeline is **multi-step orchestration**. It calls multiple services in sequence (and/or in parallel within a step), threads their outputs together, manages phase boundaries (eviction hints to the worker), and surfaces progress to the UI.

Pipelines are deliberately separate from services:

- **Services** = "make one model call". They're stateless adapters around a single provider per modality.
- **Pipelines** = "produce a finished artifact". They coordinate the services, handle Streamlit progress widgets, and own the phase boundaries.

## Pipeline modules

- [cinematic.md](cinematic.md) — the main flow. Type a theme, get a finished short film: script → audio + storyboard → per-scene videos → ffmpeg merge with optional lipsync, watermark, logo. ~635 lines, the heart of the app.
- [srt.md](srt.md) — a thin re-export from `services/llm/parser.py` for callers that think of "load my custom SRT" as a pipeline concern.

## Why pipelines are Streamlit-coupled

The cinematic pipeline reads/writes `st.session_state` directly and updates progress widgets (`st.progress`, `st.status`, `st.expander`) mid-loop. That coupling is intentional for now — it lets each phase emit real-time UI updates without extra plumbing.

The downside: the pipeline isn't directly reusable from a CLI or batch job. Splitting into pure-data orchestration + thin Streamlit renderers is on the [roadmap](../roadmap.md).

## Interaction with the worker

Every model call inside a pipeline goes through `core.worker_client.worker`. Pipelines never import service facades directly. This is what makes the worker / web split work — pipelines are agnostic to whether the worker is local or remote.

```python
from core.worker_client import worker
from core.registry import session_preferred

# Tier-picker-driven model selection per call:
worker.generate_script(theme=..., model_id=session_preferred("llm"))
worker.generate_image(prompt=..., model_id=session_preferred("image"))
worker.synthesize(text=..., model_id=session_preferred("tts"))
worker.generate_video(prompt=..., model_id=session_preferred("video"))
worker.apply_lipsync(video_path=..., audio_path=..., model_id=session_preferred("lipsync"))
```

ffmpeg merging, watermark/logo overlay, BGM mixing, and audio time-stretch run **Streamlit-side** in the pipeline. They're fast, file-bound, and don't benefit from the HTTP boundary.

## Phase-boundary eviction

Pipelines call `worker.evict_models(modality=...)` between phases so the worker frees memory for the next phase's heavy weights. See [memory-management.md](../memory-management.md).

## Adding a new pipeline

If you're building a different end-to-end flow (e.g. "explainer video" with diagrams instead of cinematic stills), add a new module under `pipelines/` and a corresponding doc here. The pattern:

1. Module-level functions per phase (`generate_X`, `generate_Y`, `final_merge`).
2. Each phase reads/writes `st.session_state` for handoff.
3. Worker calls go through `core.worker_client.worker`.
4. ffmpeg / pydub work runs Streamlit-side.
5. Eviction hints at phase boundaries.
6. Wire into a new UI tab in `ui/tabs/`.

See [extending.md](../extending.md) for the full extension guide.
