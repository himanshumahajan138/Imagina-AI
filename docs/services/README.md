# Services Layer

`services/<modality>/` is the second tier in the dependency graph: `core/` < `services/` < `pipelines/`, `worker/`, `ui/`. Every model call in the system flows through this layer.

## Anatomy

Each modality directory has the same shape:

```
services/<modality>/
  __init__.py
  service.py             Facade — picks model, dynamically imports backend, dispatches
  prompts.py             (LLM only) Prompt templates
  parser.py              (LLM only) Output parsers
  backends/
    __init__.py
    <provider>.py        One file per provider; implements core/protocols.py
```

The five modalities:

- [llm.md](llm.md) — script generation
- [image.md](image.md) — cinematic stills
- [video.md](video.md) — per-scene clips
- [tts.md](tts.md) — voice synthesis
- [lipsync.md](lipsync.md) — talking-head sync
- [media.md](media.md) — ffmpeg helpers (no `service.py` — pure utility)

## The facade pattern

`services/<modality>/service.py` looks the same across modalities (modulo the protocol method name):

```python
import importlib
from core.protocols import <Modality>Backend
from core.registry import pick_model, session_preferred

_BACKEND_MODULE = "services.<modality>.backends"

class <Modality>Service:
    def __init__(self, model_id: str | None = None) -> None:
        preferred = model_id or session_preferred("<modality>")
        self.model_id, self.cfg = pick_model("<modality>", preferred=preferred)
        self._backend: <Modality>Backend = self._load_backend()

    def _load_backend(self) -> <Modality>Backend:
        mod = importlib.import_module(f"{_BACKEND_MODULE}.{self.cfg['backend']}")
        return mod.build_backend(self.cfg)

    def <protocol_method>(self, ...):
        return self._backend.<protocol_method>(...)
```

### Why a facade and not direct backend imports?

1. **Pipelines and UI never need to know which backend is active.** They call `worker.generate_image(...)` and the worker calls `ImageService(model_id=...).generate_image(...)`. Adding a new image provider doesn't touch any caller.
2. **Lazy backend imports.** `importlib.import_module` only loads the module the user actually picked. Users without `mlx-lm` installed never trigger an import error from `services/llm/backends/mlx_local.py` — it's not imported unless they pick it.
3. **Worker routes can be ~10 lines.** The route module just builds the service and calls it; there's no per-provider logic at the route layer.

### Why `_load_backend` not `getattr`-style dispatch?

Each backend module exports a `build_backend(cfg) -> <Modality>Backend` function. We could replace this with class registration (`@register("openai")`) but registries-with-decorators have load-order pitfalls — the decorator only fires if the module is imported, so you'd end up importing every backend at startup just to populate the registry.

`build_backend` keeps the loading explicit and lazy.

## Backend protocols

Defined in `core/protocols.py` ([core.md](../core.md#coreprotocolspy) for details). The five interfaces:

```python
class LLMBackend(Protocol):
    name: str
    tier: Tier
    def generate_script(self, theme, duration, language, **kwargs) -> Script: ...

class ImageBackend(Protocol):
    name: str
    tier: Tier
    def generate_image(self, prompt, out_path, dimension, reference_images=None, **kwargs) -> MediaAsset: ...

class VideoBackend(Protocol):
    name: str
    tier: Tier
    def generate_video(self, prompt, out_path, dimension, duration, seed_image=None, **kwargs) -> MediaAsset: ...

class LipsyncBackend(Protocol):
    name: str
    tier: Tier
    def apply(self, video_path, audio_path, out_path, **kwargs) -> MediaAsset: ...

class TTSBackend(Protocol):
    name: str
    tier: Tier
    def synthesize(self, text, out_path, voice, speed=1.0, language="a", **kwargs) -> MediaAsset: ...
```

Backends:
- Set `name = "<backend_name>"` and `tier = Tier.LOCAL/CLOUD_OSS/API` as class attributes.
- Take `cfg: dict[str, Any]` in `__init__` (the yaml entry).
- Implement the protocol method. Any provider-specific kwargs come through `**kwargs`.
- Raise `BackendUnavailable` when deps/env/checkpoints are missing; `GenerationFailed` when a request fails.
- Return `MediaAsset` with `path`, `kind`, and JSON-serialisable `meta`.

### Why `**kwargs`?

Universal arguments are fixed by the protocol. Provider-specific extras (OpenAI's `previous_response_id`, F5-TTS's `ref_audio`, Wav2Lip's tweaks) come through `**kwargs`. The cinematic pipeline passes everything it has; backends pick what they need and ignore the rest.

Loses static type safety on the extras. Worth it for the flexibility.

### Optional `warmup()` hook

Local backends that hold heavy weights should implement:

```python
def warmup(self) -> None:
    # Trigger the lazy load now.
    self._pipe()  # or _model_and_tokenizer(), etc.
```

`worker/main.py:_warmup_llm` calls this on startup for the auto-picked LLM. If you add a new local backend, implement `warmup` so it's preload-eligible.

## Backend taxonomy

Across all modalities, every backend falls into one of these provider templates:

| Template | Tier | Examples | What it does |
| --- | --- | --- | --- |
| OpenAI | API | gpt-image-1, Sora, GPT script | Direct OpenAI SDK calls |
| Gemini | API | Imagen, VEO 3, Gemini script | Direct google-genai SDK calls |
| Sync.so | API | Sync.so lipsync | Direct sync SDK calls (uploads via static server first) |
| ElevenLabs | API | ElevenLabs TTS | Direct elevenlabs SDK calls |
| Replicate | Cloud OSS | FLUX, Wan, Hunyuan, DeepSeek, F5-TTS, LatentSync | `core/replicate_client.run` + URL download |
| MLX local | Local | Qwen 2.5 7B (LLM only) | `mlx-lm` on Apple Silicon |
| Diffusers local | Local | SDXL-Turbo (image), Z-Image-Turbo (image), LTX-Video (video) | `diffusers` + MPS / CUDA |
| Kokoro local | Local | Kokoro TTS | `kokoro` package, fp32 on M2 |
| ONNX local | Local | Wav2Lip lipsync | `onnxruntime` + Core ML EP |

**Why these templates exist explicitly in code rather than as one giant abstraction:**
- Each provider's SDK has different conventions (sync vs async, request shape, file vs URL inputs).
- The boring shared parts of each provider live in helper modules: `core/replicate_client.py`, `server/static.py`. The unique parts live per-backend.
- Adding a new provider in an existing template is small (~50-100 lines). Adding a new template (e.g. when AWS Bedrock becomes a thing) means writing a new helper, but you don't break anything else.

## Service-level introspection

`services/video/service.py` adds non-protocol surface for the UI:

```python
class VideoService:
    @property
    def produces_audio(self) -> bool          # for lipsync skip
    @property
    def scene_duration(self) -> int           # 8 / 10 / 12 seconds
    @property
    def max_total_duration(self) -> int       # for slider max
    @property
    def default_total_duration(self) -> int   # for slider default

# module-level helper used by the sidebar
def video_constraints() -> dict[str, int]:
    return {"scene": ..., "min": ..., "max": ..., "default": ..., "step": ...}
```

The sidebar reads `video_constraints()` directly — this stays in-process (no HTTP roundtrip) because it's just yaml introspection, not a model call.

## How dispatch resolves at runtime

A request like `worker.generate_image(prompt="...", model_id="z-image-turbo")` flows:

```txt
1. ui/tabs/cinematic.py
       worker.generate_image(model_id=session_preferred("image"), ...)

2. core/worker_client.py
       POST /images/generate {prompt, model_id="z-image-turbo", ...}

3. worker/routes/images.py
       manager.submit(_run_image, req)         # async: returns 202 + job_id

4. worker thread runs _run_image(req):
       ImageService(model_id="z-image-turbo").generate_image(...)

5. services/image/service.py:__init__
       pick_model("image", preferred="z-image-turbo")
       → cfg = {"backend": "zimage_local", "hf_id": "Tongyi-MAI/Z-Image-Turbo", ...}
       importlib.import_module("services.image.backends.zimage_local")
       backend = mod.build_backend(cfg)

6. services/image/backends/zimage_local.py:generate_image(...)
       pipe = self._pipe()
            → model_manager.get("diffusers::Tongyi-MAI/Z-Image-Turbo", _load_pipeline, cost_gb=12)
            → ZImagePipeline.from_pretrained(...) on first call only
       result = pipe(prompt=..., width=..., height=..., ...)
       result.images[0].save(out_path)
       return MediaAsset(path=..., kind="image", meta={"model": ..., "steps": 9})

7. Job thread completes, MediaAsset stored on the job record.

8. Client polls GET /jobs/{id} → 200 {status: done, result: {path, kind, meta}}
       → MediaAsset reconstructed
       → returned to caller

9. pipelines/cinematic.py:_generate_storyboard_images yields the MediaAsset
       → UI renders the scene
```

Five layers between the UI button and the GPU. Each layer has one job; none of them know about the others' details.

## Service-level testing

Right now the codebase has no service-level unit tests. The smoke-test pattern we use during development:

```python
# 1. spin up the worker in a thread
import uvicorn, threading
from worker.main import app
server = uvicorn.Server(uvicorn.Config(app, port=PORT, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()

# 2. point the worker_client at it
os.environ["IMAGINA_WORKER_URL"] = f"http://127.0.0.1:{PORT}"

# 3. exercise via WorkerClient
from core.worker_client import WorkerClient
client = WorkerClient(...)
asset = client.generate_image(prompt="test", out_path="/tmp/x.png", dimension="1024x1024")
```

This works for end-to-end testing but doesn't isolate the service layer. Future: build mock backends keyed off model_id (e.g. `"_test"` backend in each modality) so service-level tests don't need real model weights.

## Future improvements

- **Backend registry validation** at `load_registry` time — verify `backend` resolves to an importable module and that `build_backend(cfg)` returns a protocol-compliant object. Fails earlier than request-time.
- **Mock backends** for testing — each modality could ship `services/<modality>/backends/_test.py` that returns deterministic outputs without real models.
- **Per-backend cost tracking** — instrument latency + bytes-out + dollars-in in the facade layer, log to `JobResult`-shaped records, surface in the UI.
- **Streaming responses** — `LLMBackend.generate_script_stream` for token-by-token UI updates. Currently every backend buffers and returns at the end.
- **Capabilities advertising** — backends could advertise `supports_reference_images: bool`, `max_prompt_tokens: int`, etc. The UI could disable "Image Refinement Mode" for backends that don't support response-id chaining instead of silently ignoring the kwarg.
