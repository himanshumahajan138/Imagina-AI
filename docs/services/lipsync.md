# Lipsync Service — Talking-Head Sync

Replaces the lip region of a video to match a separate audio track. Used in the cinematic pipeline only when (a) `use_custom_audio` is enabled, (b) `lipsync_mode` is on in the sidebar, and (c) the active video backend doesn't natively produce audio (`produces_audio=false`).

VEO 3 produces audio-synced video natively; the pipeline skips lipsync entirely when VEO is active.

## Files

```
services/lipsync/
  __init__.py
  service.py
  backends/
    __init__.py
    sync_api.py          Sync.so API (default API tier)
    replicate.py         LatentSync 1.5 via Replicate
    wav2lip_local.py     Local Wav2Lip ONNX (Core ML execution provider on macOS)
```

## Protocol method

```python
def apply(
    self,
    video_path: Path,
    audio_path: Path,
    out_path: Path,
    **kwargs: Any,
) -> MediaAsset:
```

The pipeline calls this with the merged-scenes video + the merged-scenes audio (as already-time-aligned WAVs / MP4s). Lipsync backends read both, run inference, write a new MP4 with the lip-synced video and the original audio muxed in.

## Backends

### `sync_api.py` — Sync.so (default API)

```yaml
sync-so:
  tier: api
  backend: sync_api
  requires_env: SYNC_API_KEY
```

Sync.so's API takes URLs, not file uploads. So we **upload** the video + audio to a small static file server first (`server/static.py`), let Sync.so download via those URLs, then download the result.

#### Why a static file server

Sync.so accepts `https://example.com/foo.mp4` URLs. Localhost paths obviously won't work; uploading bytes via their SDK isn't supported. Solutions:

- **Object store** (S3, GCS): operationally heavier, requires creds.
- **Self-hosted static server**: tiny FastAPI on `:8000` that serves `/files/<id>` from a tmp dir. Uses ngrok (`pyngrok`) or a public URL when behind NAT.

We chose the latter. See `server/static.py`. The `BASE_URL`, `STATIC_SERVER_PORT`, `USE_NGROK`, `NGROK_AUTH_TOKEN` env vars in `.env.example` configure it.

#### 299-second chunking

Sync.so has a 5-min input limit. For longer videos:

```python
def lipsync_generation_pipeline(video_path, audio_path):
    video_duration = get_file_duration(video_path)
    audio_duration = get_file_duration(audio_path)
    needs_split = video_duration > 299 or audio_duration > 299

    if needs_split:
        video_chunks = split_media_into_chunks(video_path, 299)
        audio_chunks = split_media_into_chunks(audio_path, 299)
        synced_chunks = []
        for (v_start, v_end, v), (a_start, a_end, a) in zip(video_chunks, audio_chunks):
            v_url = upload_file_to_static_server(v)
            a_url = upload_file_to_static_server(a)
            synced_chunk_path = sync_so_lipsync_pipeline(audio_url=a_url, video_url=v_url, ...)
            synced_chunks.append(synced_chunk_path)
        merge_videos(synced_chunks, str(merged_output_path))
        return merged_output_path

    # < 299s: single round trip
    video_url = upload_file_to_static_server(video_path)
    audio_url = upload_file_to_static_server(audio_path)
    return sync_so_lipsync_pipeline(audio_url, video_url, ...)
```

Chunking is via `services/media/merger.py:split_media_into_chunks` and `merge_videos`; both are headless (logger only, no Streamlit calls) so they work server-side.

#### Polling Sync.so

```python
response = get_client().generations.create(
    input=[Video(url=video_url), Audio(url=audio_url)],
    model="lipsync-2-pro",
    options=GenerationOptions(sync_mode="remap"),
    output_file_name="quickstart",
)
job_id = response.id

while status not in ["COMPLETED", "FAILED", "REJECTED"]:
    time.sleep(10)
    status = get_client().generations.get(job_id).status

# downloads from generation.output_url to a tempfile
```

Sync mode `"remap"` is what produces the cleanest output for our use case (full talking-head re-render rather than just mouth region).

### `replicate.py` — LatentSync 1.5 (cloud OSS)

```yaml
latentsync:
  tier: cloud_oss
  backend: replicate
  replicate_id: bytedance/latentsync
  requires_env: REPLICATE_API_TOKEN
```

Simpler than Sync.so — Replicate accepts file uploads directly. Single API call, no chunking required (LatentSync handles longer videos internally):

```python
with open(video_path, "rb") as v, open(audio_path, "rb") as a:
    output = run(self.replicate_id, input={"video": v, "audio": a})
url = first_url(output)
download(url, out_path)
```

### `wav2lip_local.py` — Wav2Lip ONNX (local)

```yaml
wav2lip:
  tier: local
  backend: wav2lip_local
  ram_gb: 1
  ckpt_path: models/wav2lip.onnx
```

Full local pipeline using ONNX Runtime + OpenCV face detection + librosa mel spectrogram + ffmpeg mux. ~150 lines of code in `wav2lip_local.py`.

#### Setup

```bash
pip install onnxruntime-coreml          # macOS; or `pip install onnxruntime` elsewhere
python -m scripts.download_models --modality lipsync   # places models/wav2lip.onnx
```

The ONNX checkpoint isn't bundled (~500 MB). Multiple Wav2Lip-ONNX repos exist with subtly different I/O shapes; we assume the canonical shape:

```
face : (N, 6, 96, 96)  float32   3×RGB current frame stacked with 3×RGB lower-half-masked frame
mel  : (N, 1, 80, 16)  float32   80 mel bins × 16 time steps
out  : (N, 3, 96, 96)  float32   lip-synced face crop, RGB
```

If your checkpoint differs, the backend raises `GenerationFailed` with a message pointing at the I/O contract.

#### Pipeline

```python
def apply(self, video_path, audio_path, out_path, **kwargs):
    session = self._session()           # ort.InferenceSession, cached

    # 1. Load video frames
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = [...]
    cap.release()

    # 2. Audio → mel chunks aligned to fps
    mel_chunks = _compute_mel_chunks(str(audio_path), fps)
    n = min(len(frames), len(mel_chunks))
    frames, mel_chunks = frames[:n], mel_chunks[:n]

    # 3. Face detect (Haar cascade — bundled in opencv-python)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # 4. Per-frame inference + paste-back
    out_frames = []
    for frame, mel in zip(frames, mel_chunks):
        face_box = _detect_face(frame, cascade)
        if face_box is None:
            out_frames.append(frame)            # no face — pass through
            continue
        x, y, w, h = face_box
        face_resized = cv2.resize(frame[y:y+h, x:x+w], (96, 96))
        face_input = _build_face_input(face_resized)[None]    # (1, 6, 96, 96)
        mel_input = mel[None, None].astype("float32")          # (1, 1, 80, 16)
        lip = session.run(None, {"face": face_input, "mel": mel_input})[0]
        # paste lip-synced 96×96 back into original frame
        out_frame = frame.copy()
        out_frame[y:y+h, x:x+w] = cv2.resize(_lip_to_bgr(lip), (w, h))
        out_frames.append(out_frame)

    # 5. Write silent video, mux audio via ffmpeg
    cv2.VideoWriter(silent_path, fourcc=mp4v, fps, (w, h)).write_all(out_frames)
    subprocess.run(["ffmpeg", ..., "-i", silent_path, "-i", audio_path,
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", out_path])
```

Cache key: `f"wav2lip::{ckpt_path}"`. The cached object is the `ort.InferenceSession`. Loading is fast (~1 s); the bulk of the time is per-frame inference.

#### Performance

ONNX with the Core ML execution provider on M2 lands ~10-20 fps inference. So a 30-second talking-head at 24 fps = 720 frames = ~36-72 s per scene plus the ffmpeg mux. Fine for casual use; not the fastest option for batch.

#### Robustness

The Haar cascade misses some faces (especially profile views or weird lighting). When it does, the frame passes through untouched — the mouth doesn't move on those frames. A more robust face detector (mediapipe, retinaface) would help; punted on the extra dependency for now.

## When the pipeline calls lipsync

In `pipelines/cinematic.py:final_generation`, after the per-scene videos are concat-merged:

```python
if use_custom_audio:
    audio_path = video_data.get("audio_path")
    native_audio = video_produces_audio()         # reads VideoService.produces_audio
    if (st.session_state.lipsync_mode
        and audio_path and Path(audio_path).exists()
        and not native_audio):
        # Run lipsync
        lipsynced_video_path = _run_lipsync(merged_tmp, audio_path)
        if lipsynced_video_path:
            # ffmpeg encode to final
        else:
            # fallback: attach audio without lipsync
```

`_run_lipsync` calls `worker.apply_lipsync` which is async — polls `/jobs/{id}` under the hood. Failures are logged and return `None`; the pipeline falls back to attaching audio without lipsync.

## Async at the worker

`POST /lipsync/apply` is async because Sync.so's polling can take 1-5 minutes per chunk and Wav2Lip ONNX is per-frame inference (also minutes for long clips).

`WorkerClient.apply_lipsync` polls every 5 s.

## Memory & eviction

Wav2Lip's ONNX session is ~500 MB. Cache key `wav2lip::`. The pipeline doesn't currently call `evict_models("lipsync")` because:

- Lipsync runs in `final_generation`, after which the cinematic flow is complete (no more model calls).
- Sync.so / LatentSync don't load anything locally.

If a future workflow chains another generation step after lipsync, add `worker.evict_models("lipsync")` at the appropriate boundary.

## Failure modes

| Failure | What's raised |
| --- | --- |
| `onnxruntime` not installed | `BackendUnavailable("onnxruntime not installed. macOS: pip install onnxruntime-coreml.")` |
| Wav2Lip checkpoint missing | `BackendUnavailable("Wav2Lip checkpoint not found at models/wav2lip.onnx ...")` |
| ONNX I/O mismatch | `GenerationFailed("Wav2Lip ONNX inference failed at frame N: ... different I/O contract.")` |
| Sync.so API error | `None` returned from `sync_so_lipsync_pipeline`; caller falls back to audio-only attach |
| Replicate URL download fails | `GenerationFailed(...)` from `core.replicate_client` |
| Static file server upload fails | `None` returned; logged; caller falls back |
| Sync.so chunk failure mid-batch | Returns `None`; partial chunks not merged |

## Static file server context

`server/static.py` is a tiny FastAPI app on a separate port (default 8000). Started outside `honcho` (it's launched by the Sync.so backend on first use, or you start it manually). When ngrok is configured, the public URL is what Sync.so reaches; on a local network you can use a LAN IP.

`upload_file_to_static_server(local_path) -> str` writes the file to the server's tmp dir and returns `f"{BASE_URL}/files/{file_id}"`. The server cleans up files after Sync.so has fetched them (or on TTL).

This is **not** part of the worker daemon — it predates the worker split. Eventually it could become a worker route (`POST /upload` returning a URL) if/when we want one less moving piece.

## Future improvements

- **Better face detection** — mediapipe or retinaface instead of Haar. Worth the extra dep for face robustness.
- **Sync.so chunking parallelism** — currently sequential. Sync.so's API allows concurrent jobs; could fan out the chunks.
- **Wav2Lip GFPGAN refinement** — many forks combine Wav2Lip with GFPGAN face restoration for cleaner output. Optional second-pass model.
- **Onnx batching** — currently per-frame inference. Stacking N frames into a batch dim (the ONNX model supports it) speeds up CPU/MPS inference.
- **Skip face detect when there's no face** — first frame's detection result could be cached and reused (or a simple optical-flow tracker between frames). Halves the cv2 work.
- **Real-time progress in the UI** — currently a single status line until lipsync finishes. The job manager could emit progress events that the client `on_progress` callback surfaces.
- **Replace static file server with worker upload route** — `POST /upload` → `{url}` would unify the moving pieces.
- **Per-frame cancellation hook** — backends could check a flag every N frames so DELETE actually halts in-flight Wav2Lip work.
- **LatentSync feature parity** — Replicate's LatentSync supports more knobs (lip_smoothing, frame_rate); not currently exposed.
