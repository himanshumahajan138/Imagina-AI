# `core/` — Reference

Shared infrastructure used by every other package. Nothing in `core/` imports from `services/`, `pipelines/`, `ui/`, or `worker/` — the dependency arrow only points inward.

## Module map

| File | What it is | Imported from |
| --- | --- | --- |
| `core/types.py` | Dataclasses: `Tier`, `ScriptBlock`, `Script`, `MediaAsset`, `JobResult` | services, pipelines, worker |
| `core/protocols.py` | Backend protocols (`LLMBackend`, `ImageBackend`, `VideoBackend`, `LipsyncBackend`, `TTSBackend`) | services |
| `core/errors.py` | `ImaginaError`, `BackendUnavailable`, `QuotaExceeded`, `ConfigError`, `GenerationFailed` | everywhere |
| `core/logger.py` | Module-level `logger` singleton | everywhere |
| `core/config.py` | Static dicts: speakers, languages, dimensions, resolutions, aspect-ratio map | UI, services, pipelines |
| `core/registry.py` | yaml loader + `pick_model` + `available_models` + `is_pickable` + `common_dimensions` + `session_preferred` | services, UI |
| `core/model_manager.py` | Process-wide LRU model cache + `evict` + GPU cache drain | local backends, worker eviction route |
| `core/replicate_client.py` | Shared helpers for Replicate-backed cloud_oss backends | services/*/backends/replicate.py |
| `core/worker_config.py` | env-knob constants (`WORKER_URL`, `SYNC_TIMEOUT_S`, etc.) | worker_client, app.py |
| `core/worker_client.py` | `WorkerClient` HTTP client + module-level `worker` singleton | pipelines, UI |
| `core/job_queue.py` | Thin `ThreadPoolExecutor` wrapper (legacy stub) | nothing live; superseded by `worker/jobs.py` |
| `core/storage.py` | (Reserved — currently empty) | nothing |
| `core/utils.py` | Back-compat re-export shim from when monolithic `utils.py` was split | only legacy imports |

## `core/types.py`

Dataclasses shared across the request lifecycle. Pure-data, no behaviour.

```python
class Tier(str, Enum):
    LOCAL = "local"
    CLOUD_OSS = "cloud_oss"
    API = "api"

@dataclass
class ScriptBlock:
    script: str
    scene: str
    video_scene: str
    start_time: str
    end_time: str

@dataclass
class Script:
    blocks: list[ScriptBlock]
    theme: str | None
    duration: int | None
    language: str | None

@dataclass
class MediaAsset:
    path: Path
    kind: str          # "image" | "audio" | "video"
    meta: dict[str, Any]

@dataclass
class JobResult:
    success: bool
    asset: MediaAsset | None
    error: str | None
    backend: str | None
    tier: Tier | None
    latency_ms: int | None
```

**Why a `Tier` enum instead of a string field?** Type-checker catches typos and you can iterate `Tier` to drive UI rendering.

**Why `MediaAsset.meta` as `dict[str, Any]`?** Backend-specific metadata leaks (OpenAI's `response_id`, diffusers `device` string, audio sample rate). The dict has to round-trip through the worker's JSON response, so values must be JSON-serialisable scalars — `str(torch.device('mps'))`, not `torch.device('mps')`. Easy to forget; if you see `PydanticSerializationError: Unable to serialize unknown type` in the worker log, that's the cause.

`JobResult` is currently unused — it's the planned shape for telemetry once we instrument latency/cost/backend tracking. Keep it for the day someone adds metrics.

## `core/protocols.py`

`Protocol` (PEP 544) interfaces that every backend implements. Decorated with `@runtime_checkable` so `isinstance(x, LLMBackend)` works in tests.

```python
@runtime_checkable
class LLMBackend(Protocol):
    name: str
    tier: Tier
    def generate_script(self, theme, duration, language, **kwargs) -> Script: ...

@runtime_checkable
class ImageBackend(Protocol):
    name: str
    tier: Tier
    def generate_image(self, prompt, out_path, dimension, reference_images=None, **kwargs) -> MediaAsset: ...

# ... VideoBackend, LipsyncBackend, TTSBackend
```

**Why protocols, not abstract base classes?** Protocols are structural — backends don't have to inherit. Importing `core.protocols.LLMBackend` is free (no torch/openai deps); inheriting `LLMBackend(ABC)` would force every backend module to import the ABC even when its protocol-method signature was already correct.

**Why `**kwargs`?** Backends have provider-specific knobs (OpenAI's `previous_response_id`, F5-TTS's `ref_audio`). The protocol-level signature carries the universal arguments; everything else passes through `**kwargs`. Yes, this loses static checking on the extras — accepted trade.

## `core/errors.py`

Three-level hierarchy:

```
ImaginaError                 (base)
├── BackendUnavailable       missing dep / env / checkpoint / network
├── QuotaExceeded            provider rate-limited or quota'd
├── ConfigError              malformed yaml / missing modality / etc.
└── GenerationFailed         backend ran but produced no usable output
```

**Convention**: anything that's "the system isn't broken, but this request can't succeed" is `ImaginaError`. Native Python exceptions (`KeyError`, `TypeError`, `ValueError` from our own code) signal bugs.

**Why this matters for the worker**: `worker/routes/*.py` catches `ImaginaError` and maps to HTTP 502. Other exceptions become 500 with a traceback in the body. The client side (`core.worker_client._raise_for_status`) re-raises 502 as `GenerationFailed`, so pipeline-level `except GenerationFailed:` works whether the backend is local or worker-side.

## `core/logger.py`

A single configured `logger` named `app`. Format: `timestamp - app - LEVEL - file:line - msg`.

```python
from core.logger import logger
logger.info(f"[diffusers-local] generating {width}x{height}")
```

**Conventions**:
- Tag log lines with the subsystem in brackets: `[mlx-llm]`, `[diffusers-local]`, `[wav2lip-local]`, `[audio-gen]`. Makes filtering easy.
- `logger.exception` for unexpected errors with traceback; `logger.error` for expected-failure messages without.
- Don't add per-message timestamps — the formatter handles it.

## `core/config.py`

Static lookup tables shared between UI and services:

- `SPEAKER_OPTIONS` — friendly name → Kokoro voice id (e.g. `"Heart" → "af_heart"`).
- `COMMON_LANGUAGES` — friendly name → Kokoro lang code (`"American English" → "a"`).
- `DIMENSIONS` — friendly label → `"WxH"` value (3 entries: landscape / portrait / square).
- `MODEL_TYPES` — legacy mapping (`"SORA" → "openai"`); only used to mirror the picked video model into `st.session_state.model_type` for back-compat.
- `RESOLUTIONS` — friendly name → ffmpeg `vf` scale string (720p / 1080p / 4k).
- `ASPECT_RATIOS` — dimension string → aspect ratio string (`"1536x1024" → "16:9"`). Used by Replicate / VEO / Imagen which prefer aspect-ratio over WxH.
- `SORA_DIMENSIONS` — our internal dimension → Sora's accepted dimension (Sora doesn't support 1024×1024 directly; we map square → 1280×720).
- `FASTWAN_DIMENSIONS` — legacy, FastWan video backend is currently commented out.

**Why static tables and not yaml?** These are tightly coupled to the registered backends (e.g. `SPEAKER_OPTIONS` keys must match Kokoro's voice strings). Adding a new voice is a code change anyway, and the type-checker catches typos.

## `core/registry.py`

Single source of truth for "which backend does the system pick when the user asks for modality X".

```python
load_registry()                      # → cached dict from configs/models.yaml
list_models(modality)                # → dict[model_id → cfg]
env_satisfied(cfg)                   # all `requires_env` keys set?
is_stub(cfg)                         # `unavailable: true` flag?
is_pickable(cfg)                     # env_satisfied AND not stub
available_models(modality)           # env-satisfied (incl. stubs, for UI)
supported_dimensions(modality, mid)  # cfg["supported_dimensions"] or None
common_dimensions(modalities)        # ∩ across active picks per modality
label_for(modality, model_id)        # "gpt-image-1 (API)" friendly label
pick_model(modality, preferred=None) # → (mid, cfg)  with tier fallback
session_preferred(modality)          # st.session_state.preferred_models[modality]
```

### `pick_model` rules

1. If `preferred` is set and its env is satisfied, return it. Stubs are honoured here — the user explicitly opted in by selecting them from the sidebar.
2. Otherwise iterate `Tier.API → Tier.CLOUD_OSS → Tier.LOCAL`, picking the first model that's `is_pickable` (env satisfied AND not a stub).
3. If nothing's pickable, raise `BackendUnavailable` with a constructed message that lists exactly which env var unlocks which model: `"Set REPLICATE_API_TOKEN → flux-dev, ..."`.

### Why the `unavailable: true` flag

Some local backends are stubs (the file imports cleanly but `generate_*` raises `GenerationFailed("...pending Phase 4")`). Without the flag, auto-pick would land on them and crash mid-pipeline with a confusing message. With the flag, auto-pick skips them but the sidebar still shows them (with a `🚧 stub` tag) so the user can manually select if they want to test.

### Dimension intersection

`common_dimensions(["image", "video"])` returns the dimension values both active models support. The sidebar uses this to filter the dimension dropdown — picking SDXL (square only) + LTX (square only) gives you only `1024x1024`. Picking gpt-image-1 (all three) + sora (all three) gives you all three.

If the intersection is empty (disjoint sets), the sidebar falls back to all of `DIMENSIONS` with a warning. See `ui/sidebar.py:_filtered_dimensions`.

### Why yaml

- Adding a backend = adding a yaml entry. No registry edits.
- Per-model metadata flows to the UI (`scene_duration` for the slider, `supported_dimensions` for the dropdown).
- Tier rules are explicit, not buried in import order.

## `core/model_manager.py`

Process-wide LRU cache for heavy local model objects. See [memory-management.md](memory-management.md) for the full deep-dive. Quick-reference here:

```python
get_manager()                                # process singleton
manager.get(key, loader, cost_gb)            # load-or-return-cached
manager.evict(key)                           # del + gc.collect + GPU cache drain
manager.evict_modality("llm")                # evict all keys with that modality's prefix
manager.evict_all()
manager.resident()                           # snapshot of current cache
```

Cache-key prefix convention: `mlx::`, `diffusers::`, `kokoro::`, `ltx::`, `wav2lip::`. Backends construct keys like `f"diffusers::{self.model_id}"`. The `_MODALITY_PREFIXES` map at the top of the file routes `evict_modality` to the right prefixes. **If you add a new local backend, pick a unique prefix and update that map.**

## `core/replicate_client.py`

Shared helpers for tier-2 (Replicate-hosted) backends. Lazy-imports the `replicate` SDK so non-cloud_oss callers don't pull it.

```python
require_token()              # raises BackendUnavailable if REPLICATE_API_TOKEN unset
run(model_ref, input)        # thin wrapper around replicate.run with clearer errors
download(url, out_path)      # streams URL to disk
first_url(output)            # normalise list/str/FileOutput → URL string
join_text(output)            # iterable[str] → joined str (LLM token streams)
```

Every Replicate backend (`services/*/backends/replicate.py`) is ~50 lines because the boring parts live here.

## `core/worker_config.py`

Env-driven knobs for the HTTP client. All read at import time.

| Constant | Env | Default | Purpose |
| --- | --- | --- | --- |
| `WORKER_URL` | `IMAGINA_WORKER_URL` | `http://127.0.0.1:8005` | Where the client sends requests |
| `SYNC_TIMEOUT_S` | `IMAGINA_WORKER_SYNC_TIMEOUT` | `600` | HTTP timeout on `/scripts`, `/tts` (covers cold MLX/Kokoro loads) |
| `DEFAULT_POLL_INTERVAL_S` | `IMAGINA_WORKER_POLL_INTERVAL` | `5` | Polling cadence for async jobs |
| `DEFAULT_JOB_TIMEOUT_S` | `IMAGINA_WORKER_JOB_TIMEOUT` | `1800` | Hard cap on a single async job |

Bumping `SYNC_TIMEOUT_S` higher is fine; it's only the upper bound, not actual wait time.

## `core/worker_client.py`

The HTTP-side mirror of the worker's API. Mirrors service-facade signatures so the pipeline migration was mechanical:

```python
worker.health()                                             # GET /health
worker.is_alive()                                           # bool, never raises
worker.models()                                             # GET /models
worker.generate_script(theme, duration, language, ...)      # → DataFrame
worker.generate_script_raw(...)                             # → Script
worker.generate_image(prompt, out_path, dimension, ...)     # async; polls; returns MediaAsset
worker.synthesize(text, out_path, voice, ...)               # → MediaAsset (sync)
worker.generate_video(prompt, out_path, ...)                # async; polls
worker.apply_lipsync(video_path, audio_path, out_path, ...) # async; polls
worker.evict_models(modality=..., key=..., all=False)       # best-effort hint
worker.get_job(job_id) / worker.cancel_job(job_id)
```

`_await_job` is the polling loop — fires `on_progress` callback per status change, raises `GenerationFailed` on terminal error or timeout.

**Module-level singleton `worker`**: pipelines and UI import this directly. Constructing a `WorkerClient` is free (no network call), but reusing the singleton keeps connection pools warm.

**Test-friendly transport**: `WorkerClient(transport=httpx.ASGITransport(app=fastapi_app))` would let unit tests bypass sockets, except `httpx.ASGITransport` is async-only. Tests that need the round-trip spin up a real `uvicorn.Server` in a thread (see existing smoke-test snippets).

**Error mapping**: 502 → `GenerationFailed`, 503 → `BackendUnavailable`, network errors → `BackendUnavailable`. Anything else → `GenerationFailed("Worker NNN: ...")` so callers don't have to disambiguate.

## `core/job_queue.py`

Thin wrapper around `ThreadPoolExecutor` from before the worker daemon existed. Currently nothing live imports it — `worker/jobs.py:JobManager` is the in-use implementation. Kept for the day we want to introduce a separate Streamlit-side queue (e.g. for ffmpeg merge running in the background).

Status: stub, safe to delete if it ever becomes confusing.

## `core/storage.py`

Empty placeholder. Reserved for the file-handoff abstraction we'll need when the worker moves to a remote machine — at that point a `StorageBackend` protocol with `upload(local_path) → url` and `download(url, out_path)` implementations for `tmp://`, `s3://`, `gcs://`, etc. would land here. Nothing to look at right now.

## `core/utils.py`

Back-compat re-export shim. Before the refactor, `utils.py` was a 700-line monolith. We split it into `services/`, `pipelines/`, and `ui/components/`; this file re-exports the public names so older notebooks or scripts that did `from core.utils import openai_image_generator` still work.

```python
__all__ = [
    "IMAGE_PROMPT", "SCRIPT_PROMPT", "WATERMARK",
    "crop_image_to_dimension", "extract_frame",
    "final_generation", "gemini_image_generator",
    "gemini_video_generation_pipeline", "generate_audio_images",
    "generate_video", "hash_df", "lipsync_generation_pipeline",
    "logo_addition", "merge_videos", "normalize_veo3_video",
    "openai_image_generator", "openai_script_generator",
    "parse_script_scene_content", "remove_watermark_ffmpeg",
    "remove_watermark_opencv", "save_uploaded_file",
    "sora_video_generation_pipeline", "split_media_into_chunks",
    "storyboard_gallery", "sync_so_lipsync_pipeline",
    "validate_script_data", "video_gallery", "watermark_addition",
]
```

New code should import from canonical homes; the shim is for not-yet-migrated callers.

## Future improvements

- **`storage.py`** — implement when remote-worker becomes real.
- **Probe-based `cost_gb`** — `model_manager.get` could measure RSS delta on load and update its bookkeeping, instead of trusting the yaml estimate.
- **Telemetry hook** — wire `JobResult` into a metrics sink (Prometheus, OTLP) so we can see latency/cost per backend in aggregate.
- **`utils.py` removal** — if nothing imports from it for a release cycle, delete the shim.
- **Better protocol assertions** — currently a backend that forgets `name: str` causes a runtime `AttributeError`; a `runtime_checkable` instance check at backend-build time would surface the issue earlier.
