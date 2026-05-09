# Model Registry & `models.yaml`

The yaml file at [configs/models.yaml](../configs/models.yaml) is the **single source of truth** for which providers exist and which the system picks at runtime. Every other layer reads from it: the worker auto-pick, the sidebar tier picker, the duration slider, the dimension dropdown.

This doc describes the schema, the picker's selection rules, and how the UI reflects per-model metadata.

## Schema

Top-level structure:

```yaml
<modality>:                       # llm | image | video | lipsync | tts
  default_tier: auto              # currently unused; kept for future "force tier X" override
  models:
    <model_id>:                   # arbitrary string; UI displays it as-is
      tier: local | cloud_oss | api
      backend: <module_name>      # filename in services/<modality>/backends/, no .py
      requires_env: KEY1          # str or list[str]; all must be set for env_satisfied
      # … modality-specific fields below
```

Five modalities, each with the same shape. Adding a model never requires registry code changes — just yaml.

### Universal fields

| Field | Type | Purpose |
| --- | --- | --- |
| `tier` | `local` / `cloud_oss` / `api` | Picker's tier ranking. API > Cloud OSS > Local. |
| `backend` | string | Maps to `services/<modality>/backends/<backend>.py`. Module must export `build_backend(cfg) -> <Modality>Backend`. |
| `requires_env` | string or `[string]` | All keys must be present in env for `env_satisfied()` to return True. Optional. |
| `unavailable: true` | bool | Stub flag — `is_pickable()` returns False. Sidebar shows it with `🚧 stub` tag. Auto-pick skips. |
| `supported_dimensions` | `[string]` | Restricts the sidebar dimension dropdown intersection. Optional; absence means "all". |

### Per-modality extras

**LLM:**
- `hf_id` (MLX / HF backends): HuggingFace repo id of the weights.
- `replicate_id` (Replicate): provider model reference.
- `model` (OpenAI / Gemini): API-side model name.

**Image:**
- `hf_id`, `replicate_id`, `model` as above.
- `ram_gb` (local): rough memory cost for the LRU budget.
- `num_inference_steps`, `guidance_scale`, `vae_id` (SDXL backend specifically): quality knobs surfaced via cfg.
- `cpu_offload` (Z-Image): toggle `enable_model_cpu_offload()`.

**Video:**
- `hf_id`, `replicate_id`, etc.
- `ram_gb` for local.
- `scene_duration` (int seconds): per-scene clip length the backend emits — drives the duration slider min/step in the sidebar and the audio fit target.
- `produces_audio` (bool, default false): set to true for VEO 3 — the pipeline skips lipsync when the video backend already produces audio-synced output.
- `fps`, `num_inference_steps` (LTX local).

**Lipsync:**
- `replicate_id`, `ckpt_path` (local).

**TTS:**
- `hf_id`, `replicate_id`, `model`.
- `ram_gb` for local.

## Worked example

```yaml
image:
  default_tier: auto
  models:
    sdxl-turbo:                          # listed first → preferred within local tier
      tier: local
      backend: coreml_local
      hf_id: stabilityai/sdxl-turbo
      ram_gb: 8
      num_inference_steps: 4             # quality knob
      guidance_scale: 0.0
      # vae_id: madebyollin/sdxl-vae-fp16-fix   # optional override
      supported_dimensions: ["1024x1024"]

    z-image-turbo:
      tier: local
      backend: zimage_local
      hf_id: Tongyi-MAI/Z-Image-Turbo
      ram_gb: 12
      num_inference_steps: 9
      cpu_offload: false
      supported_dimensions: ["1024x1024", "1024x1536", "1536x1024"]

    flux-dev:
      tier: cloud_oss
      backend: replicate
      replicate_id: black-forest-labs/flux-dev
      requires_env: REPLICATE_API_TOKEN

    imagen-3:
      tier: api
      backend: gemini
      requires_env: GOOGLE_GENAI_API_KEY

    gpt-image-1:
      tier: api
      backend: openai
      requires_env: OPENAI_API_KEY
```

## How the picker decides

Code: `core/registry.py:pick_model` ([core/registry.py:75](../core/registry.py#L75) onwards).

```python
def pick_model(modality, preferred=None) -> (model_id, cfg):
    # 1. Honour explicit user preference if env is satisfied.
    #    Stubs are honoured — the user opted in by selecting from sidebar.
    if preferred and preferred in models and env_satisfied(models[preferred]):
        return preferred, models[preferred]

    # 2. Otherwise iterate API → CLOUD_OSS → LOCAL.
    #    Within each tier, take the first model that's pickable
    #    (env satisfied AND not a stub).
    for tier in (Tier.API, Tier.CLOUD_OSS, Tier.LOCAL):
        for mid, cfg in models.items():
            if Tier(cfg["tier"]) == tier and is_pickable(cfg):
                return mid, cfg

    # 3. Nothing pickable. Raise BackendUnavailable with a constructed
    #    message that lists which env var unlocks which backend.
    raise BackendUnavailable(_no_backend_message(modality, models))
```

### Tier order rationale

API > Cloud OSS > Local because:
- API is paid, so the user explicitly opted in by configuring the key — they almost certainly want it used.
- Cloud OSS (Replicate) is paid-per-use too, also explicit.
- Local is the "no keys configured" fallback. It's the safety net, not the preferred default for a fully-configured user.

If you want to invert this for a specific deployment (e.g. force local-first to save money), add a `prefer_tier` knob to the picker. Currently we don't support it because nobody asked.

### Why preferred can override tier

A user with `OPENAI_API_KEY` set might still want to use the local Z-Image for image gen (it's free + private). The sidebar tier picker writes `st.session_state.preferred_models["image"] = "z-image-turbo"`, `session_preferred("image")` reads that, and `pick_model` honours it.

If a user picks something whose env *isn't* satisfied (e.g. `gpt-image-1` without `OPENAI_API_KEY`), the picker silently falls back to the auto-pick. There's no error UI for "your selection isn't reachable" yet — the sidebar greys it with `⚠ env not set` so the user can see why.

## The `unavailable: true` stub flag

Some local backends are scaffolded but their actual ML inference is unimplemented (the backend module imports cleanly, but `generate_*` raises `GenerationFailed`). Without the flag, auto-pick would happily land on them and crash mid-pipeline.

With the flag:
- `is_pickable(cfg)` returns False → auto-pick skips.
- `is_stub(cfg)` returns True → sidebar adds `🚧 stub` to the dropdown label.
- User can still manually select; the backend will fail at request time but that's the user's choice.

Currently `unavailable: true` is **not set on any model**. SDXL, Z-Image, LTX, Wav2Lip, and Kokoro are all real implementations now. The flag exists for future stubs and for users who want to forcibly hide a backend without removing its yaml entry.

## `supported_dimensions` and the dimension dropdown

The sidebar's "Video Dimensions" selectbox is the **intersection** of `supported_dimensions` across the active image and video models, in the order they appear in `core.config.DIMENSIONS`.

Code: [ui/sidebar.py:_filtered_dimensions](../ui/sidebar.py#L18) calls `core.registry.common_dimensions(["image", "video"])`.

Examples:
- `gpt-image-1` (no constraint) + `sora` (no constraint) → all three (`Landscape`, `Portrait`, `Square`).
- `sdxl-turbo` (square only) + any video → only `Square`.
- `sdxl-turbo` (square only) + `ltx-video-2b` (square only) → `Square`.
- Hypothetical: image-only-portrait + video-only-landscape → empty intersection. The sidebar warns and falls back to all of `DIMENSIONS` so the UI doesn't soft-lock.

**Why intersect**: the sidebar is one global dropdown; whatever the user picks must work for both the image and the per-scene video. Listing dimensions only one of them can do would lead to mid-generation aspect-ratio surprises.

## `scene_duration` and the duration slider

Each video model declares its native chunk length:
- `veo-3`: 8 s
- `sora`: 12 s
- everything else: 10 s

The sidebar's "Duration (seconds)" slider reads this via `services/video/service.py:video_constraints()` → `{scene, min, max, default, step}`.

Default min/max/step:
- `min = scene_duration`
- `max = scene_duration × 15`
- `default = scene_duration × 2`
- `step = scene_duration`

So picking VEO 3 gives you `8s, 16s, 24s, ..., 120s`. Picking Sora gives `12s, 24s, ..., 180s`.

The same value drives the audio fit target — each TTS scene is force-fit to exactly `scene_duration` seconds. See [audio-pipeline.md](audio-pipeline.md).

## `produces_audio` and lipsync skip

VEO 3 emits video with native synced audio (same shot includes the talking-head and the lip-sync, no separate audio track needed). The cinematic pipeline checks `worker.produces_audio()` (which reads this flag) and skips the lipsync phase when it's true — saves a costly Sync.so round trip and avoids the double-sync artefacts.

If you add a future video backend with this property, set `produces_audio: true` in yaml. Nothing else needs to change.

## Per-modality cache key prefix (memory eviction)

Each local backend uses a unique cache-key prefix when calling `model_manager.get(...)`:

| Modality | Prefix | Used by |
| --- | --- | --- |
| LLM | `mlx::` | `services/llm/backends/mlx_local.py` |
| Image | `diffusers::` | `coreml_local.py` (SDXL), `zimage_local.py` (Z-Image) |
| Video | `ltx::` | `services/video/backends/ltx_local.py` |
| Lipsync | `wav2lip::` | `services/lipsync/backends/wav2lip_local.py` |
| TTS | `kokoro::` | `services/tts/backends/kokoro_local.py` |

These prefixes are mapped to modalities in `core/model_manager.py:_MODALITY_PREFIXES` so `evict_modality("image")` knows which keys to drop.

**If you add a new local backend**, pick a unique prefix and add it to that map. See [memory-management.md](memory-management.md).

## yaml ordering matters

Within a single tier, `pick_model` iterates in yaml insertion order and picks the first env-satisfied non-stub. So **yaml order = preference order within tier**.

We use this to set the local-tier default for image:

```yaml
image:
  models:
    sdxl-turbo:        # listed first → preferred local default
      tier: local
      ...
    z-image-turbo:     # also local; only auto-picked if sdxl-turbo isn't pickable
      tier: local
      ...
```

If you want Z-Image as the local default instead, swap them. No code change.

## Adding a new model

Three steps. Always.

1. **Implement** `services/<modality>/backends/<name>.py` against the matching `core/protocols.py` interface. Export `build_backend(cfg) -> <Modality>Backend`.
2. **Register** in `configs/models.yaml`:
   ```yaml
   <modality>:
     models:
       my-model:
         tier: api          # or cloud_oss / local
         backend: my_name   # = filename without .py
         requires_env: MY_API_KEY
         supported_dimensions: ["1024x1024"]   # if applicable
   ```
3. **Done.** No registry edits, no pipeline edits, no UI edits. The sidebar tier picker discovers it; the worker routes to it via `<Modality>Service(model_id=...)`.

For local backends with weight caching, also implement `warmup()` and pick a unique cache prefix. See [extending.md](extending.md).

## Future improvements

- **`prefer_tier` global** — let an env var force tier ordering for a deployment (e.g. `IMAGINA_PREFER_TIER=local` for a fully-offline deploy).
- **Validation on load** — `load_registry` could check that each `backend` actually resolves to an importable module and a valid protocol implementation. Right now you find out at request time.
- **Per-model `cost_gb` probe** — the yaml estimate is rough. Memory manager could record actual RSS delta on load and write back to a runtime cache.
- **Schema versioning** — once we have downstream tools that consume the yaml shape, version the schema with a `schema_version: 1` top-level field and a migration path.
- **Surface "you picked an unreachable model" in UI** — currently silent fallback. Could show an info banner: "Your image picker (gpt-image-1) is unavailable (env not set); auto-picked sdxl-turbo instead."
