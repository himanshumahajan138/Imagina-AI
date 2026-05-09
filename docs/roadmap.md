# Roadmap & Known Limitations

A living list of things we know are imperfect and would improve given time. Pick one when you're looking for impactful work.

Organised by area. Severity guesses (S = small / 1-3 hrs, M = medium / 1-2 days, L = large / week+) are rough — every estimate is wrong.

## Architecture

| Item | Severity | Notes |
| --- | --- | --- |
| **Pure-data orchestration** in `pipelines/cinematic.py` | M | Currently every phase reads/writes `st.session_state` and emits Streamlit widgets mid-loop. Splitting into `pure(...)` data functions + thin Streamlit renderers would let the same pipeline run from a CLI or batch job. |
| **Janitor for `/tmp/imagina/`** | S | Both processes leak temp files. A periodic sweeper that TTL-purges old job dirs (1 hr default) keeps disk usage bounded. |
| **Storage abstraction** (`core/storage.py`) | M | Today both processes share `/tmp`. When the worker moves to a different machine (remote GPU), file handoff has to switch to byte uploads or a shared object store. The empty `core/storage.py` is the placeholder. |
| **Persistent job manager** | M | `worker/jobs.py:JobManager` is in-memory; a worker restart loses pending jobs. Redis + RQ would persist across restarts and enable multi-worker. |
| **Resumable pipelines** | M | If Streamlit restarts mid-flow, `editable_images` / `scene_videos` are gone. Persisting these to a small SQLite store would let users resume. |

## Worker daemon

| Item | Severity | Notes |
| --- | --- | --- |
| **Cooperative cancellation** | M | `DELETE /jobs/{id}` only cancels queued (not running) jobs because backends don't poll a "should-cancel" flag. Adding cancel-checks every N diffusion steps / N video frames would make `DELETE` actually halt in-flight work. |
| **Streaming progress** | M | Async jobs report `running` then `done` — no granularity. Adding step-level progress (`progress: 0.6, message: "step 18/30"`) would let the UI show real progress bars. Implement via Server-Sent Events on `/jobs/{id}` instead of polling. |
| **`/metrics` endpoint** | S | Prometheus exporter for request latency / success rate / cache hit rate. ~30 lines on top of the existing request lifecycle. |
| **Auth + TLS** | M | Worker is localhost-only. The moment it leaves localhost it needs auth (token or JWT) and TLS. |
| **Graceful shutdown** | S | honcho SIGINT cuts in-flight jobs. A shutdown hook that waits for the job pool to drain (with a deadline) would prevent partial work loss. |
| **Multi-worker scale-out** | L | `JobManager` is single-process. Multiple workers behind a load balancer would need shared queue (Redis), shared cache eviction signals, and per-worker GPU pinning. Big project. |

## Memory management

| Item | Severity | Notes |
| --- | --- | --- |
| **Probe-based `cost_gb`** | S | yaml estimates are inherited from model cards and rough. The manager could measure RSS delta on load and self-correct. Closer numbers = better LRU decisions. |
| **Predictive warmup** | M | Kick off load of phase-N+1 model in a background thread while phase-N is running. Risks doubling peak RAM during overlap; punted for M2 16 GB. Worth it on bigger boxes. |
| **Concurrency-friendly `get`** | S | The lock is held for the full `loader()` duration. With `max_workers > 2` this serialises loads. A per-key sentinel pattern would let concurrent loads of different keys proceed in parallel. |
| **`/models/preload` endpoint** | S | Explicit warmup trigger from the UI ("Preload Z-Image now"). Easy add. |
| **`/tmp/imagina/` janitor** | S | Same as above — disk hygiene alongside RAM. |

## Registry & yaml

| Item | Severity | Notes |
| --- | --- | --- |
| **`prefer_tier` global override** | S | Env var to force tier ordering for a deployment (e.g. `IMAGINA_PREFER_TIER=local` for fully-offline). |
| **Validation on yaml load** | S | Verify that each `backend` resolves to an importable module and that `build_backend(cfg)` returns a protocol-compliant object. Catch typos at startup, not at request time. |
| **Schema versioning** | S | Add `schema_version: 1` so we can evolve the yaml shape with a migration path. |
| **Surface "you picked an unreachable model"** | S | Sidebar greys env-unsatisfied options but generation just silently auto-picks the next-best. A mid-workflow info banner would help. |

## Per-modality

### LLM

- **Pass `scene_duration` directly** instead of legacy `model_type` for the prompt's `seconds` placeholder. (S)
- **Streaming script generation** — yield blocks as parsed instead of buffering. Useful for long scripts. (M)
- **Per-scene prompts as separate calls** — regenerate one block without re-rolling the rest. (M)
- **Few-shot examples for non-English** — quality drops outside English; few-shot in system prompt would help. (S)
- **JSON-schema validation** before `row_to_block` — better error messages than `KeyError`. (S)

### Image

- **IP-Adapter for SDXL** — unlock reference-image conditioning on the local backend. ~200 MB extra weights. (M)
- **Refinement-mode UX** — currently the toggle is on regardless of backend; could disable / rename ("OpenAI: keep style consistent") when picked backend doesn't support response-id chaining. (S)
- **Scheduler selection** — DPMSolver++, Euler-A, etc., as a yaml knob. (S)
- **Per-backend cache prefix** — would allow holding both SDXL and Z-Image resident if RAM permits (does not on M2 16 GB; useful on bigger boxes). (S)
- **Apple Core ML SDXL** — actual ANE-compiled inference. ~2-3× speedup on M2. Requires per-resolution compiled model packages. (L)
- **NSFW / safety hooks** — currently disabled by default; surface as a yaml flag. (S)

### Video

- **LTX text-to-video** — `LTXPipeline` (no image conditioning). Add as `services/video/backends/ltx_t2v_local.py`. (S)
- **Sora prompt enhancement** — Sora's API has built-in prompt enhancement we don't expose. (S)
- **VEO 3 Fast** — separate yaml entry for the cheaper variant. (S)
- **Frame interpolation** — RIFE / FILM post-pass to double fps without doubling generation cost. (M)
- **Resolution upscaling** — generate at 768×768, upscale to 1024 with a separate pass. Halves cost on local LTX. (M)
- **Delete `fastwan_local.py`** — dead code. (S)

### TTS

- **F5-TTS proper integration** — sidebar input for "reference voice clip" + per-scene optional `ref_audio`. Currently F5 is selectable but unusable end-to-end. (M)
- **Streaming TTS** — chunk synthesis as the LLM emits the script. (M)
- **Per-scene voice override UI** — voice column already exists; surface a per-scene voice selector with audio previews. (S)
- **Sample rate consistency** — Force uniform 24 kHz across backends. (S)
- **Voice library refresh** — Static `SPEAKER_OPTIONS` will go stale. yaml or registry-driven voice list. (M)

### Lipsync

- **Better face detection** — mediapipe or retinaface instead of OpenCV Haar. Worth the extra dep for robustness. (S)
- **Sync.so chunking parallelism** — concurrent jobs instead of sequential. (S)
- **Wav2Lip GFPGAN refinement** — second-pass face restoration for cleaner output. (M)
- **ONNX batching** — per-frame inference today; batching N frames would speed up CPU/MPS. (S)
- **Real-time progress in UI** — single status line until lipsync finishes; emit per-frame / per-chunk events. (M)
- **Replace static file server with worker upload route** — `POST /upload` → URL. Eliminates a moving piece. (S)
- **Per-frame cancel hook** — DELETE actually halts in-flight Wav2Lip work. (S)

### Audio pipeline

- **Crossfade between scenes** — current concat has hard cuts. 100-200 ms crossfades smooth speedup'd boundaries. (S)
- **Per-scene BGM mood** — single global BGM today; per-scene picker from a library. (M)
- **Voice-aware speed estimation** — Kokoro voices have different baselines; per-voice `wps` calibration. (S)
- **Audio normalisation** — RMS-normalise each segment before concat for consistent BGM mix volume. (S)
- **Mute scenes** — `mute` column in data editor. (S)

### Media (ffmpeg)

- **Hardware encoding** — `-c:v h264_videotoolbox` on M2 is much faster than libx264. Currently software for portability. (S)
- **`MediaService` facade** if/when we want remote encoding. (M)
- **Real progress reporting** — parse ffmpeg `-progress pipe:1` for frame-level feedback. (M)
- **Move static file server into worker** — `POST /upload` route. (S)

## UI

| Item | Severity | Notes |
| --- | --- | --- |
| **Typed `SessionState` dataclass** | S | Catch typos statically. Currently dict access scattered everywhere. |
| **Cancel button** for in-flight pipelines | S | Calls `worker.cancel_job(...)`. UX-meaningful even before backend cancellation lands. |
| **Per-scene regen progress** | S | Currently `st.spinner`; could match the structured progress of the main flow. |
| **Settings persistence** | S | Save sidebar settings to `/tmp/imagina/session.json` so refresh doesn't reset. |
| **Worker-down recheck button** | S | One-click instead of full reload. |
| **Tab routing via URL** | M | `st.navigation` for deep-linkable tabs. |
| **Theme toggle** | S | `st_theme` integration is partial. |
| **Multi-user namespacing** | M | If serving multiple users: per-user `/tmp/imagina/<user_id>/`. |

## Testing

| Item | Severity | Notes |
| --- | --- | --- |
| **Real unit tests** | M | Currently zero. Service-level + parser tests would catch most regressions. |
| **Mock backends** | S | Per modality: `services/<modality>/backends/_test.py` returning deterministic outputs. Lets service tests not need real model weights. |
| **CI** | M | GitHub Actions running pytest + a smoke-test that boots the worker and hits `/health`. |
| **Snapshot tests for prompts** | S | The `SCRIPT_PROMPT` template is hand-tuned; regression tests would catch silent quality drops. |

## Documentation

- **`utils.py` removal** — once nothing imports from `core.utils` for a release cycle, delete the back-compat shim. (S)
- **Per-tab tutorial** — a `docs/tutorials/` with step-by-step walkthroughs of merge / watermark / trimmer / youtube tabs. (M)
- **Architecture decision records (ADRs)** — for major decisions (worker split, tier picker single-source). (M)
- **Contribution guide** — `CONTRIBUTING.md` with PR conventions, testing expectations. (S)

## Ideas

These haven't been triaged into the list above yet, just thoughts:

- Replace the cinematic "wizard" with a single page that auto-runs each phase as inputs become valid. Less clicking.
- Native stem support: separate music / vocals / SFX channels in the final mix.
- Aspect-ratio-aware Ken Burns effect on storyboard images for scenes where motion is minimal.
- Image-prompt → DSL: have the LLM emit a structured scene description (subject, action, mood, lighting) instead of free-text.
- Multi-scene video models (Wan-VACE, Sora storyboarding) — leave style consistency to the model.
- Local lipsync on the M2 ANE via Apple's compiled Core ML face/lip detection — 5-10× faster than ONNX.
- Edit-during-render: regenerate a single scene's video while the rest are still queued.
- Public-link sharing: upload final to S3 + emit a shareable link from the UI.

## Pinned issues / known bugs

None at the moment that block normal use. If you find one, note it here with a date and a steps-to-reproduce.

---

When you take something on, move it from this doc into a real GitHub issue (or PR) so it's actionable + visible. The roadmap is the wishlist; the issue tracker is the work.
