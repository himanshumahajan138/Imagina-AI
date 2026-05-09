# Extending the Codebase

How to add new things — providers, modalities, UI tabs — without touching shared infrastructure. The whole architecture is designed so the common cases are mechanical.

## Mental model

Three orthogonal axes you can extend:

1. **New provider for an existing modality** (most common) — e.g. a new image model.
2. **New modality** (rare) — e.g. background-music gen, or 3D model gen.
3. **New UI tab** (orthogonal) — a standalone tool that doesn't go through the worker.

Plus two cross-cutting things:

4. **New backend template** — when a provider doesn't fit the existing OpenAI / Gemini / Replicate / local templates.
5. **Service-level cross-cutting features** — caching, retries, logging, metrics.

## 1. New provider for an existing modality

The 90% case. Three steps. Always.

### Step 1: write the backend

Pick the modality and the right template:

| Provider type | Existing template to copy |
| --- | --- |
| OpenAI-shaped API | `services/<modality>/backends/openai.py` |
| Gemini-shaped API | `services/<modality>/backends/gemini.py` |
| Replicate-hosted OSS | `services/<modality>/backends/replicate.py` |
| Apple Silicon local LLM | `services/llm/backends/mlx_local.py` |
| Local diffusers (image/video) | `services/image/backends/coreml_local.py` or `services/video/backends/ltx_local.py` |
| Local ONNX | `services/lipsync/backends/wav2lip_local.py` |

Copy the closest template, change the provider-specific bits, keep the protocol method intact. ~50-150 lines.

```python
# services/image/backends/myprovider.py
from __future__ import annotations
from pathlib import Path
from typing import Any

from core.errors import BackendUnavailable, GenerationFailed
from core.logger import logger
from core.types import MediaAsset, Tier


class MyProviderImageBackend:
    name = "myprovider"
    tier = Tier.API                                # or CLOUD_OSS / LOCAL

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.api_key = cfg.get("api_key_env", "MYPROVIDER_API_KEY")
        # ... read tunables from cfg

    def generate_image(
        self,
        prompt: str,
        out_path: Path,
        dimension: str,
        reference_images: list[Path] | None = None,
        **kwargs: Any,
    ) -> MediaAsset:
        try:
            import myprovider_sdk
        except ImportError as e:
            raise BackendUnavailable("myprovider_sdk not installed: pip install myprovider") from e

        try:
            result = myprovider_sdk.generate(prompt=prompt, ...)
            result.save(out_path)
        except Exception as e:
            raise GenerationFailed(f"MyProvider image generation failed: {e}") from e

        return MediaAsset(
            path=Path(out_path),
            kind="image",
            meta={"provider": "myprovider"},      # JSON-serialisable scalars only
        )


def build_backend(cfg: dict[str, Any]) -> MyProviderImageBackend:
    return MyProviderImageBackend(cfg)
```

Rules:
- `name` and `tier` as class attributes.
- `build_backend(cfg)` is the dispatch entry point — required.
- Lazy-import provider SDKs inside the methods that need them. Importing the module shouldn't fail when the SDK is missing.
- Raise `BackendUnavailable` for missing deps / env / checkpoints. `GenerationFailed` for actual run failures.
- Return `MediaAsset` with JSON-serialisable `meta` (no `torch.device` objects).
- For local backends with heavy weights: implement `warmup()`, use `core.model_manager.get_manager().get(key, loader, cost_gb)`, pick a unique cache prefix.

### Step 2: register in yaml

```yaml
image:
  models:
    my-new-image-model:
      tier: api          # or cloud_oss / local
      backend: myprovider     # = filename without .py
      requires_env: MYPROVIDER_API_KEY
      supported_dimensions: ["1024x1024"]    # optional; restricts dropdown
```

That's it. Within the same tier, **yaml insertion order = preference order** (auto-pick takes the first env-satisfied non-stub).

### Step 3: nothing else

- No registry edits. The picker discovers your model on next yaml load.
- No pipeline edits. Pipelines call `worker.generate_image(model_id=session_preferred("image"))`; if the user picks your model from the sidebar, it's used.
- No UI edits. The tier picker auto-includes your model.
- No worker route edits. The route passes through to `ImageService(model_id=...)` which dispatches to your backend.

### When you need extra config

Add to the yaml entry, read in `__init__`:

```yaml
my-new-image-model:
  tier: api
  backend: myprovider
  requires_env: MYPROVIDER_API_KEY
  num_inference_steps: 30
  guidance_scale: 5.0
  custom_endpoint: https://api.myprovider.com/v2/...
```

```python
def __init__(self, cfg: dict[str, Any]) -> None:
    self.cfg = cfg
    self.steps = int(cfg.get("num_inference_steps", 25))
    self.guidance = float(cfg.get("guidance_scale", 7.5))
    self.endpoint = cfg.get("custom_endpoint", "https://default.api/v1")
```

The yaml is the source of truth — anything you put there flows down to the backend.

### Adding a local backend

Two extras beyond the basic template:

**Cache prefix in `model_manager`**:

```python
# core/model_manager.py
_MODALITY_PREFIXES = {
    "llm": ("mlx::",),
    "image": ("diffusers::", "myprovider-img::"),    # ← add yours
    ...
}
```

Without this, `evict_modality("image")` won't see your cache entries. `evict()` (exact key) still works.

**`warmup()` method**:

```python
def warmup(self) -> None:
    # Trigger the lazy load now (used by worker startup preload).
    self._pipe()        # or whatever your loader is
```

The worker's startup hook only preloads the auto-picked LLM today, but `warmup()` is also called by future `/models/preload` endpoints and useful for explicit admin warming.

## 2. Add a new modality

The 5% case. You're adding e.g. background-music gen or 3D model gen.

### Step 1: protocol

```python
# core/protocols.py
@runtime_checkable
class MusicBackend(Protocol):
    name: str
    tier: Tier
    def generate_music(
        self,
        prompt: str,
        out_path: Path,
        duration: float,
        **kwargs: Any,
    ) -> MediaAsset: ...
```

### Step 2: facade

```python
# services/music/service.py
import importlib
from core.protocols import MusicBackend
from core.registry import pick_model, session_preferred

_BACKEND_MODULE = "services.music.backends"

class MusicService:
    def __init__(self, model_id: str | None = None) -> None:
        preferred = model_id or session_preferred("music")
        self.model_id, self.cfg = pick_model("music", preferred=preferred)
        self._backend: MusicBackend = self._load_backend()

    def _load_backend(self) -> MusicBackend:
        mod = importlib.import_module(f"{_BACKEND_MODULE}.{self.cfg['backend']}")
        return mod.build_backend(self.cfg)

    def generate_music(self, prompt, out_path, duration, **kwargs) -> MediaAsset:
        return self._backend.generate_music(
            prompt=prompt, out_path=out_path, duration=duration, **kwargs,
        )
```

### Step 3: yaml section

```yaml
music:
  default_tier: auto
  models:
    musicgen-medium:
      tier: cloud_oss
      backend: replicate
      replicate_id: meta/musicgen
      requires_env: REPLICATE_API_TOKEN
```

### Step 4: implement at least one backend

```python
# services/music/backends/replicate.py
# ... follows the Replicate template
```

### Step 5: tier_picker

Add to `_MODALITIES`:

```python
# ui/components/tier_picker.py
_MODALITIES = [
    ("llm", "📝 Script (LLM)"),
    ("image", "🖼️ Image"),
    ("video", "🎬 Video"),
    ("lipsync", "👄 Lip-sync"),
    ("tts", "🗣️ TTS"),
    ("music", "🎵 Music"),     # ← new
]
```

### Step 6: worker — schemas, route, client

Add a Pydantic shape:

```python
# worker/schemas.py
class MusicRequest(BaseModel):
    prompt: str
    out_path: str
    duration: float
    model_id: str | None = None
```

Add a route module:

```python
# worker/routes/music.py
from fastapi import APIRouter
from worker.jobs import manager
from worker.schemas import JobAccepted, MusicRequest
from services.music.service import MusicService

router = APIRouter(prefix="/music", tags=["music"])

def _run_music(req: MusicRequest):
    return MusicService(model_id=req.model_id).generate_music(
        prompt=req.prompt, out_path=Path(req.out_path), duration=req.duration,
    )

@router.post("/generate", response_model=JobAccepted, status_code=202)
def generate_music(req: MusicRequest) -> JobAccepted:
    return JobAccepted(job_id=manager.submit(_run_music, req))
```

Wire into `worker/main.py`:

```python
from worker.routes import ..., music
app.include_router(music.router)
```

Add a client method:

```python
# core/worker_client.py
def generate_music(
    self, prompt, out_path, duration,
    model_id=None, poll_interval=DEFAULT_POLL_INTERVAL_S,
    timeout=DEFAULT_JOB_TIMEOUT_S, on_progress=None,
) -> MediaAsset:
    accepted = self._post("/music/generate", prompt=prompt,
                          out_path=str(out_path), duration=duration,
                          model_id=model_id)
    return self._await_job(accepted["job_id"], poll_interval, timeout, on_progress)
```

### Step 7: pipeline integration

Update `pipelines/cinematic.py` (or wherever) to call `worker.generate_music(...)`. Add the appropriate phase + eviction (`worker.evict_models("music")` if local).

That's the full lap. ~5 small files touched, no infra changes.

## 3. New UI tab

Trivially scoped — no worker, no registry, no protocol.

```python
# ui/tabs/my_tool.py
import streamlit as st

def render() -> None:
    st.title(":wrench: My Tool")
    uploaded = st.file_uploader("Input", type=["mp4"])
    if not uploaded: return
    if st.button("Process"):
        with st.spinner("..."):
            result = my_logic(uploaded)
        st.video(result)
```

```python
# app.py
from ui.tabs import my_tool
tab1, ..., tab6 = st.tabs([..., "🔧 My Tool"])
with tab6: my_tool.render()
```

If your tool wraps an existing service that's worker-routed, call `worker.<method>` like the cinematic tab does.

## 4. New backend template

When a provider doesn't fit the existing templates (OpenAI / Gemini / Replicate / local). Examples: AWS Bedrock, Together AI, Anthropic, your own internal API.

Two parts:

**A. Shared helper module**, similar to `core/replicate_client.py`:

```python
# core/bedrock_client.py
def require_creds(): ...
def invoke(model_id, body): ...
def get_response(...): ...
```

**B. Per-modality backends** that use the helper:

```python
# services/llm/backends/bedrock.py
from core.bedrock_client import invoke
class BedrockLLMBackend: ...
```

```python
# services/image/backends/bedrock.py
from core.bedrock_client import invoke
class BedrockImageBackend: ...
```

Each backend file is small (~50 lines); the boring shared bits live in the helper.

## 5. Service-level cross-cutting features

When you want a feature that applies to all backends in a modality (e.g. retries, caching, telemetry):

**Wrap inside the service facade**, not at the backend level:

```python
# services/image/service.py
class ImageService:
    def generate_image(self, prompt, out_path, ...):
        for attempt in range(self._max_retries):
            try:
                return self._backend.generate_image(prompt=prompt, ...)
            except (TimeoutError, ConnectionError) as e:
                if attempt == self._max_retries - 1: raise
                time.sleep(2 ** attempt)
                continue
```

Why facade-level? Backends shouldn't know about retry policy — that's an orchestration concern. Wrapping at the facade keeps backends focused on the provider call.

For request-level metrics:

```python
class ImageService:
    def generate_image(self, prompt, out_path, ...):
        t0 = time.monotonic()
        try:
            asset = self._backend.generate_image(prompt=prompt, ...)
            telemetry.record(modality="image", model=self.model_id,
                             latency_ms=int((time.monotonic() - t0) * 1000),
                             status="ok")
            return asset
        except Exception as e:
            telemetry.record(modality="image", model=self.model_id,
                             latency_ms=int((time.monotonic() - t0) * 1000),
                             status="error", error=str(e))
            raise
```

Same shape for every modality; one helper module keeps it DRY.

## What to NOT extend

A few things look extensible but are intentionally hardcoded:

- **`core.config.DIMENSIONS`** — three options (square, portrait, landscape). Adding more sizes works on the picker side but most backends only support specific sets, leading to dimension-intersection becoming awkward. Only add if you have a backend that actually needs a fourth.
- **`core.config.SPEAKER_OPTIONS`** — Kokoro-tied. Other TTS backends remap from this to their own voice IDs (see ElevenLabs backend). If a new TTS provider has truly unique voice names, add a per-modality voice-listing endpoint instead of growing this dict.
- **The five tier-picker modalities** — `_MODALITIES` in `tier_picker.py`. Adding a sixth IS supported (see "new modality" above), but be honest about whether it's a separate concept or just another image-generation provider in disguise.

## Common gotchas

- **Forgetting `build_backend`** → `AttributeError` at backend load time. Every backend module must export it.
- **`MediaAsset.meta` with non-JSON values** → `PydanticSerializationError` at `/jobs/{id}` response time. `str(...)` your torch devices, paths, enums.
- **Cache-key collisions** — two backends sharing a prefix means `evict_modality` evicts both. Keep prefixes specific.
- **Streamlit calls in server-reachable code** — anything reachable from `services/` paths that the worker calls (image / video / TTS / lipsync backends, `services/media/` headless helpers) must NOT call `st.*`. Use `core.logger.logger` instead.
- **Heavy imports at module top-level** — `import torch` at the top of a local backend forces all torch-less callers to install torch. Lazy-import inside the loader function.
- **Forgetting `requires_env`** in the yaml means the picker thinks the model is always available. Failures show up at runtime instead of at picker-time.

## Testing your new backend

The repo's smoke-test pattern (no formal unit-test infra yet):

```python
# 1. Spin up the worker in a thread
import threading, uvicorn, os, socket
from worker.main import app
sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
os.environ["IMAGINA_WORKER_URL"] = f"http://127.0.0.1:{port}"
server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()

# 2. Reload worker_client to pick up the new URL
import importlib, core.worker_config, core.worker_client
importlib.reload(core.worker_config); importlib.reload(core.worker_client)
from core.worker_client import worker

# 3. Exercise via WorkerClient
asset = worker.generate_image(prompt="test", out_path="/tmp/x.png",
                              dimension="1024x1024", model_id="my-new-model")
print(asset)
```

Future: real unit tests with mock backends. See `services/README.md` future improvements.
