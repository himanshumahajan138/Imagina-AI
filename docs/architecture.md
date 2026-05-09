# Architecture

The big picture: what runs where, how requests move through the system, and the decisions that shape every subsystem.

## TL;DR

- **Two long-running processes**: a Streamlit web app (`app.py`, port 8004) and a FastAPI worker daemon (`worker/`, port 8005), supervised by `honcho`.
- **Worker owns model calls.** Heavy weights (MLX, diffusers, ONNX) live in the worker process so Streamlit reruns don't reload them and long jobs don't block the UI thread.
- **Streamlit owns UI + ffmpeg.** Concat/merge/watermark/audio time-stretch run locally because they're fast, file-bound, and the UI thread orchestrates them anyway.
- **Single source of model selection** is the sidebar tier picker (`ui/components/tier_picker.py`), which writes into `st.session_state.preferred_models[modality]`. The registry reads that, falls back through tiers, and constructs the right backend.
- **All backends speak protocols** defined in `core/protocols.py`. Adding a provider = one file in `services/<modality>/backends/` + one yaml entry. No registry edits, no pipeline edits.

## Process map

```txt
┌──────────────────────────────────────┐         HTTP            ┌────────────────────────────────────────┐
│  Streamlit web (app.py)              │  ─────────────────────► │  imagina-worker (worker/main.py)       │
│  port 8004                           │   POST /scripts/gen     │  port 8005                             │
│                                      │   POST /images/gen ──┐  │                                        │
│  ─ ui/sidebar.py                     │   POST /videos/gen ──┤  │  ─ services/llm/* (auto-pick + load)   │
│    └ tier_picker writes              │   POST /lipsync/apply┤  │  ─ services/image/*                    │
│      st.session_state.preferred_     │   POST /tts/synth    │  │  ─ services/video/*                    │
│      models[modality]                │   GET  /jobs/{id} ◄──┘  │  ─ services/lipsync/*                  │
│                                      │   POST /models/evict    │  ─ services/tts/*                      │
│  ─ ui/tabs/cinematic.py              │   GET  /health          │                                        │
│    └ pipelines/cinematic.py          │                         │  ─ core/model_manager (singleton)      │
│      └ core.worker_client.worker     │                         │    ├ resident: dict[key → (obj, GB)]   │
│                                      │                         │    ├ LRU eviction under budget         │
│  + ffmpeg (merge/concat)             │                         │    └ GPU cache drain on evict          │
│  + watermark/logo overlay            │                         │                                        │
│  + audio time-stretch / silence pad  │                         │  ─ worker/jobs.py                      │
│  + BGM mixing (pydub)                │                         │    └ ThreadPoolExecutor(max_workers=2) │
│                                      │                         │      for /videos, /images, /lipsync    │
└──────────────────────────────────────┘                         └────────────────────────────────────────┘
                                  shared filesystem: /tmp/imagina/<job_id>/...
```

## Data flow: a single cinematic generation

The user types a theme, hits "Generate Script", then "Generate Scenes", then "Generate Final Video", then "Final Merge". Each step:

Eviction happens at **phase entry** (start of the next phase), not phase exit. This way the LLM/image/TTS/video models stay resident through user iteration within a phase (regenerate the script, edit a scene, regen one image) and only get freed when the user clicks the next-phase button.

1. **Theme → script (LLM)**
   - UI: `ui/tabs/cinematic.py` → `_generate_script_task`
   - Calls `worker.generate_script(theme, duration, language, model_id=session_preferred("llm"))`
   - Worker `routes/scripts.py` → `LLMService(model_id=...).generate_script` → backend
   - LLM stays resident through script iteration (regen, edits, alternate SRT upload).

2. **Script → audio (TTS) + per-scene images (Image)**
   - Pipeline: `pipelines/cinematic.py` → `generate_audio_images`
   - **Phase entry: `evict_models("llm")`** to free MLX before TTS+image weights load.
   - For each scene: pre-estimate `tts_speed` from word count, call `worker.synthesize`, then run the simple "short=silence pad / long=speedup" fit so each segment lands at exactly `scene_duration`. Concatenate into one merged WAV.
   - For each scene: call `worker.generate_image` (async — see below). Reference images uploaded in Streamlit are persisted to `/tmp` first because the worker has no in-process Streamlit context.
   - Image + TTS stay resident through storyboard iteration (regen single scene, replace image).

3. **Per-scene videos**
   - Pipeline: `generate_video`
   - **Phase entry: `evict_models("image")` + `evict_models("tts")`** to free SDXL/Z-Image + Kokoro before the (heavier) video model loads.
   - For each scene calls `worker.generate_video` (async, polled). Each scene's seed image is the storyboard image from step 2.
   - Video model stays resident through video gallery iteration.

4. **Final merge (Streamlit-side)**
   - `final_generation`
   - **Phase entry: `evict_models("video")`** to free local video weights before merge/lipsync.
   - Runs ffmpeg concat-demuxer over the per-scene MP4s.
   - If lipsync is enabled and the video backend doesn't natively produce audio (`produces_audio: false`), it calls `worker.apply_lipsync` (async). LatentSync / Sync.so internally chunk if needed.
   - Final pass attaches audio (or attaches via ffmpeg `-map`), normalises codec, writes the final MP4, sets `st.session_state.final_output_path`.

## Why two processes

A single Python process is the obvious "simpler" choice. We rejected it because:

1. **Streamlit reruns evict caches.** Every time the user interacts with a widget, Streamlit re-runs the script top-to-bottom. Module-level model state survives, but if the process restarts (manual reload, hot-reload during dev, error) the MLX / diffusers / ONNX weights all reload. With heavy local backends (~12 GB Z-Image, ~7 GB SDXL), that's minutes of cold start every time.

2. **Long-running generation freezes the UI thread.** A 5-min video generation done synchronously means the Streamlit script can't yield. The user can't see status updates, can't cancel, can't even close a sidebar.

3. **One Python env is fragile.** Streamlit + diffusers + MLX-LM + onnxruntime + Replicate + ffmpeg-python + librosa + opencv all together is a hard `requirements.txt` to keep stable. Splitting lets the worker eventually carry only generation deps; the Streamlit env stays light.

The split costs us complexity (two processes to supervise, file handoff over `/tmp`, schema duplication via Pydantic). We took the trade.

See [worker.md](worker.md) for the daemon-specific details and [docs/_archive/worker-service-plan.md](_archive/worker-service-plan.md) (if present) for the original implementation plan.

## Why service facades

Every modality has a `services/<modality>/service.py` facade and a `services/<modality>/backends/<name>.py` adapter per provider. Pipelines and UI import only the facade.

### What facades buy us

- **One call site, multiple providers.** `LLMService.generate_script(...)` works the same whether the auto-pick lands on MLX, Replicate, or OpenAI. Backend lookup happens lazily (`importlib.import_module`) so heavy deps stay lazy too.
- **The worker can wrap them.** `worker/routes/scripts.py` is ~10 lines because it just builds the service and calls it. Without facades, every route would re-implement provider dispatch.
- **Streamlit reads them for introspection.** `ui/sidebar.py` calls `services.video.service.video_constraints()` to populate the duration slider — no HTTP roundtrip needed for static yaml data.

### Why protocols live in `core/`

`core/protocols.py` defines the five interfaces (`LLMBackend`, `ImageBackend`, etc.). Backends don't import each other; they only depend on `core`. This keeps the dependency graph a tree (services and pipelines depend on core; everything else is sibling).

## Why the tier picker is the single source of truth

Earlier iterations had two competing model-selection mechanisms:

- A "Model Type" sidebar selectbox (`MODEL_TYPES = {"SORA": "openai", "VEO": "gemini"}`) that mapped to a string `st.session_state.model_type`. Various backends and pipelines read this directly with magic strings.
- The new tier-picker dict `st.session_state.preferred_models[modality]`.

Both routes existed simultaneously, with subtle precedence rules. Result: changing image backend in one place didn't change it in another, and the duration slider read `model_type` while the actual generator read `preferred_models`.

We collapsed it to one source. `model_type` is now derived from `preferred_models["video"]` for legacy callers (e.g. `core/config.py` constants tables), but no decision is ever made from it.

## File-handoff strategy

On the same box (current single-machine deploy), Streamlit and the worker share `/tmp`. Streamlit writes uploads/intermediates there; the worker reads them by absolute path.

This keeps the wire format trivial: requests are JSON with paths, not byte arrays. A 12-MB scene image doesn't need to be base64-encoded over HTTP twice. The worker also writes its outputs to `/tmp/imagina/<job_id>/...` so Streamlit can read them by path on completion.

**Tradeoff**: the moment the worker moves to a different machine (remote GPU), the assumption breaks. We'll need to switch to byte uploads or a shared object store. That's the deferred-work line in the sand — see [roadmap.md](roadmap.md).

**Cleanup**: nothing automatic right now. Both processes leak temp files. A janitor pass that deletes `/tmp/imagina/*` on a TTL would be a small, isolated improvement.

## Memory budget on M2 16 GB

- Streamlit + Python + libs: ~3-5 GB
- Worker process Python: ~500 MB
- One local model resident at a time:
  - MLX Qwen 2.5 7B Q4: ~6 GB
  - SDXL-Turbo: ~7 GB
  - Z-Image-Turbo: ~12 GB
  - LTX-Video 2B: ~12 GB peak
- macOS itself: ~3-4 GB

Two heavy local models simultaneously OOM. We avoid this with **phase-aware eviction**: pipelines call `worker.evict_models(modality=...)` between phases, and the manager does `del + gc.collect() + torch.mps.empty_cache() + mlx.clear_cache()`. See [memory-management.md](memory-management.md).

## Async vs sync endpoints

| Endpoint | Mode | Why |
| --- | --- | --- |
| `POST /scripts/generate` | sync | LLM cold-load is ~30 s, generation ~5 s. Fits within the 600 s sync timeout. |
| `POST /tts/synthesize` | sync | Kokoro is ~1 GB, fast. ~2-5 s per segment. |
| `POST /images/generate` | **async** | First-run weight download is multi-minute. Subsequent ~10 s. Sync would be either too generous on timeout or cliff-fail on first run. |
| `POST /videos/generate` | **async** | LTX local is 3-6 min/clip; Sora and VEO API calls are 1-3 min. Always long. |
| `POST /lipsync/apply` | **async** | Per-scene-chunked sync.so or per-frame ONNX inference. |

Async endpoints submit to `worker/jobs.py:JobManager` (a `ThreadPoolExecutor(max_workers=2)`), return `202 + {job_id}`, and the `core.worker_client.WorkerClient` polls `/jobs/{id}` under the hood — pipeline callers never see the job_id, they just block until `MediaAsset` comes back.

## Error model

- All backends raise `core.errors.GenerationFailed` for runtime errors and `core.errors.BackendUnavailable` when their deps / env / checkpoints are missing.
- Worker routes map both to HTTP 502 with the message in the JSON body.
- `WorkerClient._raise_for_status` re-raises 502 as `GenerationFailed` so pipeline-level `except GenerationFailed:` blocks work the same whether the backend is in-process or remote.
- Network errors (worker down, timeout) raise `BackendUnavailable`. The Streamlit startup health-check banner uses this to surface a "worker down" warning.

## Where things lazy-load

- **Backend modules** are imported in `services/<modality>/service.py:_load_backend` via `importlib.import_module`. So if `mlx-lm` isn't installed, only `services.llm.backends.mlx_local` fails to import; nothing else cares.
- **Heavy deps** (torch, diffusers, onnxruntime, mlx_lm) are imported inside `_load_pipeline` / `_ensure_runtime` / `_model_and_tokenizer`. Importing the backend module doesn't pull them.
- **Model weights** load on first `*Service(...).generate_*(...)` call. The `core.model_manager` caches the loaded object across requests so the second call is instant.
- **Worker startup preload** (when `IMAGINA_PRELOAD_LLM=true`) spawns a daemon thread that fires the auto-picked LLM's `warmup()` so the first script gen lands without the cold-start tax.

## Cross-cutting concerns

- **Logging**: every module imports `from core.logger import logger`. Format is `timestamp - app - LEVEL - file:line - msg`. Both processes log to the same logger config (configurable via env in `core/logger.py`).
- **Errors**: anything that's "generation failed but not a bug" is `ImaginaError` or its subclass. Bugs propagate as native exceptions.
- **Settings**: yaml for model registry, env vars for runtime knobs (`IMAGINA_WORKER_*`, provider keys). No mutable global config object — keeps things stateless.

## Future improvements

See [roadmap.md](roadmap.md) for the full list. The structural ones:

- Move file handoff to a small object store (or HTTP byte uploads) when the worker moves to a remote box.
- Replace the in-process `JobManager` with Redis + RQ once we have multiple workers.
- Probe-based `cost_gb` updates (measure RSS delta on load, not the yaml estimate).
- Smarter predictive warmup that overlaps phase N+1 model load with phase N inference.
- A janitor for `/tmp/imagina/*`.
- Real auth + TLS when the worker leaves localhost.
