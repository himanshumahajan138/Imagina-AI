# Memory Management

How the worker keeps heavy local model weights resident across requests without OOM-ing on a 16 GB M2.

## Why this needs explicit attention

Two competing pressures:

1. **Loading is slow.** MLX Qwen 2.5 7B Q4 takes ~30 s to load. SDXL-Turbo's first run downloads ~7 GB then takes ~6 s to load. Z-Image-Turbo is ~12 GB resident from a 33-GB on-disk repo. We don't want to pay this every request.
2. **Two heavy models don't fit.** MLX Qwen (~6 GB) + SDXL (~7 GB) is fine. MLX Qwen + Z-Image (~12 GB) is right at the unified-memory ceiling on M2 16 GB once you factor in Streamlit, Python, ffmpeg, and macOS itself. LTX-Video (~12 GB peak during inference) plus anything else is OOM.

A simple LRU cache solves the first; without explicit eviction it doesn't reliably solve the second because PyTorch holds the GPU allocator pool even after `del model`.

## The pieces

| Piece | File | Purpose |
| --- | --- | --- |
| `ModelManager` | `core/model_manager.py` | Process-wide LRU + active eviction + GPU cache drain |
| `_MODALITY_PREFIXES` | same file | Maps modality → cache-key prefix(es) for `evict_modality` |
| `warmup()` per-backend | `services/*/backends/<local>.py` | Trigger weight load explicitly (used by startup preload) |
| `POST /models/evict` | `worker/routes/models.py` | HTTP entry-point pipelines call at phase boundaries |
| `worker.evict_models(...)` | `core/worker_client.py` | Best-effort client wrapper |
| Phase-boundary calls | `pipelines/cinematic.py`, `ui/tabs/cinematic.py` | Where evict actually happens |
| Startup preload | `worker/main.py:_warmup_llm` | Daemon thread that warms the LLM at worker boot |

## `ModelManager`

Single instance per process, accessed via `core.model_manager.get_manager()`.

```python
class ModelManager:
    def __init__(max_resident_gb=12.0):
        self._max_gb = max_resident_gb
        self._loaded: OrderedDict[str, tuple[obj, cost_gb]] = OrderedDict()
        self._lock = Lock()

    def get(key, loader, cost_gb) -> obj:
        # cached → bump to MRU and return
        # not cached → evict-until-fits → loader() → store
    def evict(key) -> bool                    # exact-key drop
    def evict_modality(modality) -> int       # all keys with that prefix
    def evict_all() -> int
    def resident() -> dict[key → cost_gb]     # cache snapshot
```

### Cache keys

Backends construct cache keys with a modality-specific prefix:

```python
# services/llm/backends/mlx_local.py
key = f"mlx::{self.hf_id}"

# services/image/backends/coreml_local.py
key = f"diffusers::{self.model_id}"
# (and includes vae_id when set, so VAE swaps don't reuse stale pipelines)

# services/image/backends/zimage_local.py
key = f"diffusers::{self.model_id}"   # shares prefix with SDXL — only one image model resident at a time

# services/video/backends/ltx_local.py
key = f"ltx::{self.model_id}"

# services/lipsync/backends/wav2lip_local.py
key = f"wav2lip::{self.ckpt_path}"

# services/tts/backends/kokoro_local.py
key = f"kokoro::{lang_code}"
```

`_MODALITY_PREFIXES` maps these to `evict_modality(modality)`:

```python
_MODALITY_PREFIXES = {
    "llm":     ("mlx::",),
    "image":   ("diffusers::",),
    "tts":     ("kokoro::",),
    "video":   ("ltx::",),
    "lipsync": ("wav2lip::",),
}
```

So `evict_modality("image")` drops both SDXL and Z-Image (both share `diffusers::`). That's intentional — the image-modality cache holds at most one model at a time.

**If you add a new local backend, pick a unique prefix and update `_MODALITY_PREFIXES`.** Otherwise `evict_modality` won't see it.

### `get` — load-or-return-cached

```python
def get(self, model_id, loader, cost_gb):
    with self._lock:
        if model_id in self._loaded:
            self._loaded.move_to_end(model_id)             # MRU bump
            return self._loaded[model_id][0]
        self._evict_until_fits_locked(cost_gb)             # may LRU-evict
        obj = loader()                                     # actual model load
        self._loaded[model_id] = (obj, cost_gb)
        return obj
```

The lock is held for the full duration of `loader()`. That's intentional: it prevents two threads from double-loading the same model. Trade-off: a loader that takes 5 minutes blocks all other `get` calls. In practice the worker's job pool is `max_workers=2` and two-at-a-time loads against a 16 GB budget would OOM anyway, so blocking is the safer behaviour.

### `_evict_until_fits_locked` — LRU under budget

```python
current = sum(c for _, c in self._loaded.values())
while current + incoming_gb > self._max_gb and self._loaded:
    key, (obj, cost) = self._loaded.popitem(last=False)    # oldest first
    del obj
    current -= cost
if evicted:
    gc.collect()
    _drain_gpu_caches()
```

The drain happens once at the end of the eviction batch (not per-key) since `gc.collect()` is non-trivial.

### `evict` and `evict_modality`

`evict(key)` is the public surface. After dropping from the dict it does `del obj`, `gc.collect()`, and `_drain_gpu_caches()`. The drain is **the part that actually returns memory to the OS** — without it, PyTorch keeps its allocator arena and RSS doesn't drop.

`evict_modality(modality)` looks up the prefix(es) for that modality and calls `evict` for each matching key. Returns the count.

`evict_all()` clears the dict, then drains. Used by `POST /models/evict` with `{all: true}`.

## `_drain_gpu_caches`

```python
def _drain_gpu_caches() -> None:
    # PyTorch MPS / CUDA
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # MLX
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()                       # newer MLX
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()                 # legacy
    except ImportError:
        pass
```

Lazy-imported so `model_manager` works whether or not torch / MLX are installed. Failures in the drain are logged at debug level and swallowed — eviction always continues.

**Why both `empty_cache` and `synchronize`** for MPS: `empty_cache` returns memory to the kernel; `synchronize` ensures any in-flight kernels finish before we declare memory freed. Skipping `synchronize` occasionally produces "ghost" RSS that drops a few seconds later.

## Phase-aware eviction in the cinematic pipeline

The pipeline knows the phase order: script → audio + image → video → lipsync + merge. **Eviction happens at the START of the next phase, not the end of the current one.** This way the user can iterate within a phase (regenerate the script three times, regenerate a single scene's image, etc.) without paying reload cost between iterations — the model stays warm until the user actually commits to moving forward.

```python
# pipelines/cinematic.py:generate_audio_images — at function entry
worker.evict_models(modality="llm")           # frees ~6 GB before TTS+image load

# pipelines/cinematic.py:generate_video — at function entry
worker.evict_models(modality="image")          # frees ~7-12 GB before video model loads
worker.evict_models(modality="tts")            # frees ~1 GB

# pipelines/cinematic.py:final_generation — at function entry
worker.evict_models(modality="video")          # frees local video weights before merge/lipsync
```

`worker.evict_models` is best-effort: HTTP failures log a warning and return zero. The pipeline never blocks on it.

### Why phase-entry, not phase-exit

Consider the flow:

1. User generates a script.
2. User edits a few rows in the data editor.
3. User regenerates the script (didn't like it).
4. User uploads a different SRT file instead.
5. User edits some more.
6. User finally clicks "Generate Scenes and Audio".

With **phase-exit eviction** (evict at the end of `_generate_script_task` and `_load_script_task`), steps 3-5 each evict-then-reload the LLM. Wasted ~30 seconds of reload time per iteration, three or four times.

With **phase-entry eviction** (evict at the start of `generate_audio_images`), the LLM stays warm through steps 1-5 and is only freed at step 6 — exactly when we know the user has committed to leaving the script phase. Same memory outcome, much less thrashing.

The same logic applies to per-scene image/audio iteration (regen single scene from storyboard gallery) and per-scene video regen (regen single scene from video gallery): eviction happens when the user moves to the *next* phase, not when each individual phase function returns.

### What if a phase is skipped

When a user uploads a script (skipping LLM gen), `evict_models("llm")` at the entry to `generate_audio_images` is a no-op — the LLM was never loaded. Same for an API-tier video pick: `evict_models("video")` at the entry to `final_generation` is a no-op because nothing under the `ltx::` cache prefix was ever stored.

The eviction calls are idempotent and free. Safer to always issue them than to track which phases actually loaded models.

## Startup LLM preload

`worker/main.py:_warmup_llm` runs in a daemon thread on startup:

```python
def _warmup_llm() -> None:
    svc = LLMService()
    if svc.cfg.get("tier") != "local":
        return  # nothing to preload for API/cloud_oss
    warmup = getattr(svc._backend, "warmup", None)
    if warmup is None:
        return
    warmup()                                   # blocks ~30s for MLX Qwen
```

The thread is daemon so it doesn't keep the worker alive on shutdown. It's started from the FastAPI `startup` event after the route table is built, so `/health` is already serving while weights load.

`IMAGINA_PRELOAD_LLM=false` skips it. Useful when iterating on the worker without exercising the LLM.

### Why only LLM, not image/video/TTS

- LLM is small (~6 GB), the user almost always uses it first (script gen).
- Image (12 GB Z-Image, 7 GB SDXL) is too heavy to preload — likely evicts the LLM you just preloaded.
- Video (12 GB LTX) same.
- TTS (~1 GB) loads in ~3 s — preload buys nothing.

If your deploy uses different defaults (e.g. API LLM but local image), the script's `worker.evict_models(modality="llm")` is a no-op (nothing to evict, registry didn't load weights), and the next phase's local image gen pays its own cold-load cost on first run. We don't proactively preload image because of the OOM risk.

## A typical run timeline

```txt
T=0      worker startup
T=0      _warmup_llm thread starts loading MLX Qwen (in background)
T=0      /health responding, banner clears
T=30     MLX Qwen resident (~6 GB)

T=60     user clicks "Generate Script"
T=60     /scripts/generate hits cache, ~5s generation
T=65     pipeline calls worker.evict_models("llm") → MLX freed (~6 GB returned)

T=70     user clicks "Generate Scenes"
T=70     audio gen: worker.synthesize x N scenes
                Kokoro loads on first call (~3s, ~1 GB)
                ~2-5s per segment after that
T=80     image gen: worker.generate_image x N scenes
                SDXL loads on first call (~6s, ~7 GB)
                ~25s per scene at 4 inference steps
T=180    pipeline calls evict_models("image") + evict_models("tts")
                → SDXL + Kokoro freed

T=185    user clicks "Generate Final Video"
T=185    video gen: worker.generate_video x N scenes
                LTX loads on first call (~10s, ~12 GB) — fits because we evicted
                ~3-6 min per scene
T=20min  pipeline calls evict_models("video")

T=20min  user clicks "Final Merge"
T=20min  Streamlit-side ffmpeg merge (no model competition)
T=20min  optional lipsync via Sync.so (API call, no local weights)
T=21min  final mp4 written
```

Note the headroom: at every phase boundary, the working set is one heavy model + Kokoro at most. Never two heavyweights simultaneously.

## Failure modes

- **`evict()` after the model is gone**: returns `False`, no error. Idempotent.
- **`evict_modality()` for a modality with no resident models**: returns `0`. No-op.
- **Worker down when pipeline calls `evict_models`**: `WorkerClient.evict_models` logs warning and returns `{evicted: 0, resident: {}}`. Pipeline continues.
- **Backend `warmup()` raises during preload**: caught by `_warmup_llm`, logged, worker stays up. First real `/scripts/generate` will pay the cold-load cost.
- **GPU cache drain fails** (rare; happens during MLX upgrades when the API name changed): logged at debug, eviction completes anyway. Keys are removed from the cache dict but kernel-level memory may not be reclaimed until the model is fully GC'd.

## Tuning knobs

- `ModelManager(max_resident_gb=...)` — currently hard-coded to 12.0 in `get_manager()`. Could be env-driven.
- `IMAGINA_PRELOAD_LLM=false` — skip startup LLM warmup.
- Per-model `ram_gb` in yaml — affects budget bookkeeping. Closer to real numbers = better LRU decisions.

## Future improvements

- **Probe-based `cost_gb`**: measure RSS delta on load and update bookkeeping. The yaml estimates are inherited from the model card; reality varies.
- **Predictive warmup**: kick off load of phase-N+1 model in a background thread while phase-N is running. Risks doubling peak RAM during the overlap; punted on M2 16 GB.
- **Concurrency-friendly cache**: `get` could release the lock during `loader()` if we add a per-key load-in-progress sentinel. Useful when we move to `max_workers > 2`.
- **`/models/preload` endpoint**: explicit warmup trigger from the UI ("warm up image gen now"). Easy to add.
- **Per-budget eviction policy**: today LRU. Could be cost-aware (evict big-and-cold first) or recency-tier-aware (always keep TTS warm if it's small enough to spare).
- **Cross-process eviction signals**: if we ever run multiple workers behind a load balancer, a Redis pub-sub channel could propagate eviction hints.
- **Janitor for `/tmp/imagina/<job_id>/`**: the manager's job is RAM, but disk also accumulates. A separate sweeper could TTL-purge old job dirs.
