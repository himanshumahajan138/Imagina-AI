# Worker Daemon

The `worker/` package is a FastAPI HTTP service that owns every model call. The Streamlit web app talks to it through `core.worker_client.WorkerClient`. Two processes share `/tmp/imagina/` for file handoff; everything else moves over JSON.

This doc describes the as-built daemon. The original implementation plan (kept for historical context) is in [_archive/worker-service-plan.md](_archive/worker-service-plan.md) if it survives in your tree.

## Why split it out

Three concrete pains the split fixes — repeated here from [architecture.md](architecture.md) for easy reference:

1. Streamlit reruns reload the script top-to-bottom; if MLX / diffusers weights live in that process, they reload on every error or hot-reload.
2. Long-running gen (5+ minute video, multi-minute first-image weight download) blocks the Streamlit script thread; the UI freezes.
3. One process means one Python env carries all SDK deps simultaneously. Splitting lets the worker carry just what it needs.

## File layout

```
worker/
  __init__.py              docstring only
  main.py                  FastAPI app factory, startup hooks, uvicorn entrypoint
  schemas.py               Pydantic request/response models for every endpoint
  jobs.py                  In-process JobManager (ThreadPoolExecutor)
  routes/
    __init__.py
    health.py              GET /health, GET /models
    scripts.py             POST /scripts/generate (sync)
    images.py              POST /images/generate (async)
    tts.py                 POST /tts/synthesize (sync)
    videos.py              POST /videos/generate (async)
    lipsync.py             POST /lipsync/apply (async)
    jobs.py                GET /jobs/{id}, DELETE /jobs/{id}
    models.py              POST /models/evict, GET /models/resident
```

## Endpoints

| Method | Path | Mode | Purpose |
| --- | --- | --- | --- |
| `GET` | `/health` | sync | Liveness + auto-pick per modality |
| `GET` | `/models` | sync | Full registry dump (yaml as JSON) |
| `GET` | `/models/resident` | sync | Currently-loaded cache snapshot |
| `POST` | `/models/evict` | sync | Drop loaded weights by modality / key / all |
| `POST` | `/scripts/generate` | sync | LLM script gen → `Script` JSON |
| `POST` | `/tts/synthesize` | sync | One audio segment → `MediaAsset` |
| `POST` | `/images/generate` | **async** | One cinematic still → `202 + {job_id}` |
| `POST` | `/videos/generate` | **async** | One scene clip → `202 + {job_id}` |
| `POST` | `/lipsync/apply` | **async** | Lipsync (incl. chunking) → `202 + {job_id}` |
| `GET` | `/jobs/{job_id}` | sync | Poll status; returns result on done |
| `DELETE` | `/jobs/{job_id}` | sync | Best-effort cancel (only effective for queued jobs) |

OpenAPI / Swagger UI: <http://localhost:8005/docs>.

### Why these and not others sync/async

- **Script + TTS** are sync because they're <60 s even on first cold load.
- **Image** is async because first-run weight download is multi-minute (SDXL is ~7 GB, Z-Image is ~33 GB). Subsequent calls are ~10 s, but the polling overhead is negligible.
- **Video + lipsync** are always long; obvious async candidates.

The sync timeout for the remaining sync endpoints is 600 s (env: `IMAGINA_WORKER_SYNC_TIMEOUT`) which covers cold MLX Qwen load (~30 s) plus the actual LLM generation (~5 s).

## `worker/main.py` — app factory + startup

```python
def create_app() -> FastAPI:
    app = FastAPI(title="Imagina Worker", ...)
    app.include_router(health.router)
    app.include_router(scripts.router)
    app.include_router(images.router)
    app.include_router(tts.router)
    app.include_router(videos.router)
    app.include_router(lipsync.router)
    app.include_router(jobs.router)
    app.include_router(models.router)

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info(f"[worker] startup | tmp={WORKER_TMP}")
        if _preload_enabled():
            threading.Thread(target=_warmup_llm, daemon=True, name="llm-warmup").start()

    return app

app = create_app()
```

### Why a daemon thread for LLM warmup

The startup hook returns immediately so `/health` is responsive. Loading MLX Qwen 2.5 7B Q4 takes ~30 s — blocking startup that long would have honcho, the user, and uvicorn's reloader all confused. Running in a thread:
- Worker is up and responsive instantly.
- Background thread loads the LLM into the manager cache.
- First `/scripts/generate` either hits the warm cache (fast) or competes with the warmup thread (also fast — `model_manager.get` is locked).

### Skipping warmup

`IMAGINA_PRELOAD_LLM=false` to disable. `_warmup_llm` also skips if the auto-picked LLM is API/cloud_oss tier (no local weights to load) or if the backend has no `warmup()` hook.

### Worker tmp dir

`WORKER_TMP = Path(os.getenv("IMAGINA_WORKER_TMP", "/tmp/imagina"))` is created on startup. Backends that don't get an explicit `out_path` from the caller use this as scratch. Currently no janitor; manual `rm -rf /tmp/imagina/*` if it grows.

## `worker/schemas.py` — request/response shapes

Pydantic models per endpoint. They mirror the service-facade signatures so migration was mechanical and the diff between "in-process call" and "HTTP call" is just `worker_client._post(...)` instead of `Service().method(...)`.

```python
class ScriptRequest(BaseModel):
    theme: str
    duration: int
    language: str
    model_id: str | None = None
    model_type: str | None = None    # legacy hint for backends that switch on it

class ImageRequest(BaseModel):
    prompt: str
    out_path: str
    dimension: str
    reference_images: list[str] = []
    script: str = ""
    video_scene: str = ""
    reference_text: str | None = None
    old_image: str | None = None
    previous_response_id: str | None = None
    model_id: str | None = None

class VideoRequest(BaseModel):
    prompt: str
    out_path: str
    dimension: str
    duration: float
    seed_image: str | None = None
    model_id: str | None = None

class TTSRequest(BaseModel):
    text: str
    out_path: str
    voice: str
    speed: float = 1.0
    language: str = "a"
    model_id: str | None = None

class LipsyncRequest(BaseModel):
    video_path: str
    audio_path: str
    out_path: str
    model_id: str | None = None

class MediaAssetOut(BaseModel):
    path: str
    kind: Literal["image", "audio", "video"]
    meta: dict[str, Any] = Field(default_factory=dict)

class JobAccepted(BaseModel):
    job_id: str

class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error", "cancelled"]
    result: MediaAssetOut | None = None
    error: str | None = None
```

### Why `model_id` lives in the request body

Streamlit's `st.session_state.preferred_models[modality]` is process-local. The worker has no Streamlit context. So every request explicitly passes the user's preferred model id; the worker passes it as `*Service(model_id=...)`, the service's facade passes it to `pick_model(preferred=...)`, and the registry honours it (or falls back).

### Why `MediaAsset.meta: dict[str, Any]`

Backend-specific metadata leaks: `response_id` for OpenAI image, `device` string for diffusers, `frames` for video. The dict is opaque on the wire — Pydantic doesn't try to validate the inner shape. Values **must be JSON-serialisable scalars**: strings, numbers, bools, lists/dicts of those. A `torch.device('mps')` will explode at response-encode time with `PydanticSerializationError: Unable to serialize unknown type`. Fix: `str(getattr(pipe, "device", "?"))`.

## `worker/jobs.py` — async job manager

```python
class JobRecord:
    job_id: str
    status: "queued" | "running" | "done" | "error" | "cancelled"
    result: Any                 # MediaAsset on done
    error: str | None
    created_at, started_at, finished_at: float
    future: concurrent.futures.Future

class JobManager:
    def __init__(max_workers=2, ttl_seconds=3600):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()
        self._ttl = ttl_seconds

    def submit(fn, *args, **kwargs) -> str:           # returns job_id
    def get(job_id) -> JobRecord | None
    def cancel(job_id) -> bool                        # only effective for queued
    def evict_all() -> int
    def _purge_stale_locked()                         # called inside submit, drops finished > TTL
```

Module-level singleton: `manager = JobManager()`. Routes import this directly.

### Why `max_workers=2`

Single-user mode. Two slots means one heavy job (e.g. video gen) can run while a light request (image gen of next scene) queues. Bumping to higher concurrency is a code change, not a config — bump when we have multi-user.

### TTL purge inside `submit`

Cleaning up old job records happens lazily on the next submit, not via a background thread. Keeps the lock simple, no scheduler needed. Trade-off: a worker that submits nothing for hours never purges. Real-world impact: zero (each cinematic run submits ~10-20 jobs).

### Cancel semantics

`Future.cancel()` only succeeds if the task hasn't started yet. Once running, neither the LLM call nor the diffusers loop is interruptible from outside. So `DELETE /jobs/{id}` is best-effort — the route returns the post-cancel status, which might still be `running`. Real cancellation requires cooperative interrupts in the backends (not implemented).

## Routes

Each route module is small. The pattern is:

```python
router = APIRouter(prefix="/<area>", tags=["<area>"])

# sync route — call the service directly
@router.post("/<verb>", response_model=<OutputModel>)
def <handler>(req: <RequestModel>) -> <OutputModel>:
    try:
        asset = <Service>(model_id=req.model_id).<method>(...)
    except ImaginaError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    return MediaAssetOut(path=str(asset.path), kind=asset.kind, meta=asset.meta)

# async route — submit to JobManager
@router.post("/<verb>", response_model=JobAccepted, status_code=202)
def <handler>(req: <RequestModel>) -> JobAccepted:
    job_id = manager.submit(_run_<verb>, req)
    return JobAccepted(job_id=job_id)

def _run_<verb>(req): return <Service>(model_id=req.model_id).<method>(...)
```

### Error mapping

| Backend raise | HTTP status |
| --- | --- |
| `BackendUnavailable` (subclass of `ImaginaError`) | `502` (treated same as `GenerationFailed`) |
| `GenerationFailed` | `502` |
| `ImaginaError` (other subclasses) | `502` |
| Anything else | `500` with `f"{type(e).__name__}: {e}"` body |

The client-side `WorkerClient._raise_for_status` re-raises 502 as `GenerationFailed` and 503 as `BackendUnavailable`, so pipeline-level `except GenerationFailed:` works whether the failure is local or worker-side.

### `routes/jobs.py` — result serialisation

`GET /jobs/{id}` calls `_serialize_result(rec.result)` which:
- Returns `None` for not-done jobs.
- Wraps `MediaAsset` → `MediaAssetOut`.
- Tolerates `dict`-shaped results that already have `path` + `kind` (forward-compat for non-asset job outputs).
- Raises `TypeError` for anything else — rare; surfaces as 500.

## Lifecycle of an async job

```txt
client                              worker (FastAPI)         worker (thread pool)
─────────                          ─────────────────         ───────────────────────
POST /videos/generate
    body: VideoRequest          → JobManager.submit(fn, req)
    ←  202 {job_id}                JobRecord(status=queued) created
                                                            ←  thread picks up
                                                                status = running
                                                                fn(req) calls
                                                                VideoService.generate_video
                                                                = pipe(...) on MPS
GET /jobs/{job_id}
    ←  {status: running}           ... blocking model call ...
                                                                status = done
                                                                result = MediaAsset
GET /jobs/{job_id}
    ←  {status: done,
         result: MediaAssetOut}
```

`WorkerClient._await_job` polls in a loop with `poll_interval=5s` (or 2s for image, where cached calls finish in ~10s) until terminal status. On `error`, raises `GenerationFailed` with the worker-side message. On `cancelled`, raises `GenerationFailed("job ... was cancelled")`.

## Memory-management routes (`routes/models.py`)

```
POST /models/evict
    body: { modality?, key?, all? }
    → { evicted: int, resident: {key: cost_gb} }

GET /models/resident
    → { key: cost_gb, ... }
```

Pipelines call `worker.evict_models(modality="llm")` after script gen, `("image")` and `("tts")` after the storyboard phase, `("video")` after the per-scene videos. See [memory-management.md](memory-management.md) for the full lifecycle rationale.

`GET /models/resident` is a debug aid — useful for "wait, why is the worker holding on to that?" investigations. No auth, localhost-bound.

## Running the worker

Three options, in order of preference:

1. **honcho** — `Procfile` starts both web and worker:
   ```
   worker: uvicorn worker.main:app --port 8005 --reload
   web:    streamlit run app.py --server.address 0.0.0.0 --server.port 8004
   ```
   Single command: `honcho start`.

2. **Two terminals** — direct uvicorn + streamlit. Useful for differentiating logs.

3. **`python -m worker.main`** — runs `worker/main.py:main()` which calls `uvicorn.run` directly. No reload. For prod-ish runs.

### Env variables

| Var | Default | Effect |
| --- | --- | --- |
| `IMAGINA_WORKER_PORT` | `8005` | Port `python -m worker.main` binds to |
| `IMAGINA_WORKER_TMP` | `/tmp/imagina` | Worker scratch dir |
| `IMAGINA_PRELOAD_LLM` | `true` | Background LLM warmup at startup |

Plus the client-side knobs in `core/worker_config.py`: `IMAGINA_WORKER_URL`, `IMAGINA_WORKER_SYNC_TIMEOUT`, `IMAGINA_WORKER_POLL_INTERVAL`, `IMAGINA_WORKER_JOB_TIMEOUT`. These are read by Streamlit, not the worker itself.

## Observability

- **Logs**: structured to the `app` logger (`core/logger.py`). Both processes log to stdout; honcho prefixes with `worker.1` / `web.1` so they're visually separable.
- **`/health`**: liveness + which model_id is auto-picked per modality. Streamlit pings this at startup to show the "worker down" banner.
- **`/models/resident`**: cheap snapshot of cached weights. Useful manually; not currently surfaced in the UI.

No metrics endpoint, no tracing. If you need them, an `/metrics` Prometheus exporter is ~30 lines on top of the request lifecycle.

## Failure modes

| Failure | What happens |
| --- | --- |
| Worker process down | Streamlit `_check_worker()` flips `worker_alive=False`, banner shows. `worker.evict_models()` and `worker.is_alive()` log + return zero/false. Generation calls raise `BackendUnavailable("Worker unreachable...")`. |
| Worker hangs | Sync calls hit `SYNC_TIMEOUT_S`; async calls hit `DEFAULT_JOB_TIMEOUT_S` and the client cancels the job. |
| Worker OOMs mid-generation | Job's worker thread either crashes (`status=error`, message in log) or the whole process dies and honcho restarts it (jobs lost). |
| Backend deps missing | `BackendUnavailable("...pip install ...")` from the backend, mapped to 502 with the actionable message in the body. |
| Bad model_id | Picker either falls back (no error) or raises `BackendUnavailable` listing required env vars. Client surfaces the message via `_raise_for_status`. |

## Future improvements

- **Probe-based eviction triggers** — the manager could refuse a load when actual RSS exceeds budget, not just declared `cost_gb`.
- **Cooperative cancel** — backends could check a `should_cancel` flag periodically (e.g. between diffusers steps) so DELETE actually halts in-flight work.
- **Streaming** — replace polling with Server-Sent Events for lower-latency progress updates. Trade-off: more middleware (auth, retries) once we leave localhost.
- **Replace in-process JobManager with Redis + RQ** when we have multiple workers.
- **Auth + TLS** when worker leaves localhost.
- **`/metrics` for Prometheus** if/when we want quantitative observability.
- **Graceful shutdown** that finishes in-flight jobs (currently honcho SIGINT cuts them).
- **Job persistence across restarts** — currently in-memory; a worker restart loses pending jobs.
- **Janitor** for `/tmp/imagina/<job_id>/` after job result is delivered.
