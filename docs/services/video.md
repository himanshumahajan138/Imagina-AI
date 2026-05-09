# Video Service — Per-Scene Clip Generation

Generates one short video clip per scene (typically 8-12 seconds) from a prompt + seed image. The cinematic pipeline concatenates these into the final movie.

## Files

```
services/video/
  __init__.py
  service.py             VideoService + scene_duration / produces_audio / video_constraints helpers
  backends/
    __init__.py
    openai.py            Sora image-to-video
    google_flow.py       VEO 3 image-to-video (produces audio natively)
    replicate.py         Wan 2.1, HunyuanVideo, any Replicate i2v model
    ltx_local.py         LTX-Video 2B local (diffusers + MPS)
    fastwan_local.py     LEGACY: FastWan local — currently entirely commented out
```

## `VideoService`

Standard facade plus modality-specific introspection:

```python
class VideoService:
    @property
    def produces_audio(self) -> bool          # cfg flag, default False
    @property
    def scene_duration(self) -> int           # cfg "scene_duration" or 10
    @property
    def max_total_duration(self) -> int       # cfg or scene_duration * 15
    @property
    def default_total_duration(self) -> int   # cfg or scene_duration * 2

def video_constraints() -> dict[str, int]:    # module-level helper for sidebar slider
    """{scene, min, max, default, step} for the duration slider."""
```

The sidebar reads `video_constraints()` directly (no HTTP roundtrip — it's just yaml introspection). The slider produces the right options for the active video model: VEO 3 → 8/16/24/.../120; Sora → 12/24/36/.../180; OSS → 10/20/30/.../150.

## Universal protocol method

```python
def generate_video(
    self,
    prompt: str,
    out_path: Path,
    dimension: str,
    duration: float,
    seed_image: Path | None = None,
    **kwargs: Any,
) -> MediaAsset:
```

Every backend takes a seed image. Text-to-video isn't exposed even where backends support it — the cinematic pipeline always generates a still first (more directable, easier to iterate on), and the still becomes the seed.

The prompt the pipeline passes is multi-line:

```
Script: <spoken line>
Image Scene: <scene description>
Video Scene: <video direction>
```

Backends can use this verbatim or extract the "Video Scene" portion if needed. Most just feed the whole thing.

## Backends

### `openai.py` — Sora 2

```yaml
sora:
  tier: api
  backend: openai
  requires_env: OPENAI_API_KEY
  scene_duration: 12
```

Sora's API is strict about output dimensions. We map our internal dimension ("1024x1024") to one of Sora's accepted sizes via `SORA_DIMENSIONS`:

```python
SORA_DIMENSIONS = {
    "1024x1536": "720x1280",     # portrait
    "1536x1024": "1280x720",     # landscape
    "1024x1024": "1280x720",     # square → defaults to landscape
}
```

The seed image is cropped to the target dimension first via `services/media/watermark.py:crop_image_to_dimension` (which despite living in `media/watermark.py` is a general-purpose centre-crop helper).

```python
video = get_client().videos.create(
    model="sora-2",
    prompt=prompt,
    size=dimension,
    input_reference=Path(cropped_image_path),
    seconds=str(duration),
    timeout=600,
)
# poll while video.status in ("in_progress", "queued")
content = get_client().videos.download_content(video.id, variant="video")
content.write_to_file(output_path)
```

`produces_audio = False` — Sora produces video only; audio is added downstream.

### `google_flow.py` — VEO 3 (audio-native)

```yaml
veo-3:
  tier: api
  backend: google_flow
  requires_env: GOOGLE_GENAI_API_KEY
  scene_duration: 8
  produces_audio: true
```

VEO 3 generates video with native synced audio — same shot includes the talking-head AND the matched lip-sync. The `produces_audio: true` flag tells the cinematic pipeline to **skip the lipsync phase** when this backend is active (saves a Sync.so / LatentSync round-trip and avoids double-syncing artifacts).

```python
operation = get_client().models.generate_videos(
    model="veo-3.0-generate-preview",
    prompt=prompt,
    image=image,
    config=genai_types.GenerateVideosConfig(
        number_of_videos=1,
        aspect_ratio=ASPECT_RATIOS[dimension],
    ),
)
# poll while not operation.done
video = operation.response.generated_videos[0]
get_client().files.download(file=video.video)
video.video.save(output_path)
```

### `replicate.py` — Wan 2.1, Hunyuan, etc.

```yaml
wan-2.1:
  tier: cloud_oss
  backend: replicate
  replicate_id: wavespeedai/wan-2.1-i2v-720p
  requires_env: REPLICATE_API_TOKEN
  scene_duration: 10

hunyuan-video:
  tier: cloud_oss
  backend: replicate
  replicate_id: tencent/hunyuan-video
  requires_env: REPLICATE_API_TOKEN
  scene_duration: 10
```

Single backend serves multiple Replicate-hosted i2v models. The input shape is the safe superset of what Wan, Hunyuan, and LTX-on-Replicate accept; Replicate ignores unknown keys:

```python
with open(seed_image, "rb") as image_file:
    output = run(self.replicate_id, input={
        "prompt": prompt,
        "image": image_file,
        "aspect_ratio": ASPECT_RATIOS.get(dimension, "16:9"),
        "num_frames": max(16, int(duration * 24)),
        "num_inference_steps": 25,
        "guidance_scale": 5.0,
    })
url = first_url(output)
download(url, out_path)
```

Adding a new Replicate i2v model is a yaml-only change.

### `ltx_local.py` — LTX-Video 2B (local)

```yaml
ltx-video-2b:
  tier: local
  backend: ltx_local
  hf_id: Lightricks/LTX-Video
  ram_gb: 12
  scene_duration: 10
  fps: 24
  num_inference_steps: 30
  supported_dimensions: ["1024x1024"]
```

Image-to-video via `LTXImageToVideoPipeline` from `diffusers`. bf16 on MPS. ~3-6 minutes per 8-second clip on M2 16 GB — slow but it works.

#### Constraints LTX cares about

```python
def _parse_dimension(dim, fallback=(768, 768)):
    w, h = ...
    return (w // 32) * 32, (h // 32) * 32         # LTX requires divisible-by-32

def _round_frames(num_frames):
    n = max(9, min(257, int(num_frames)))
    return ((n - 1) // 8) * 8 + 1                  # LTX requires (n - 1) % 8 == 0
```

So an 8-second clip at 24 fps becomes `((192 - 1) // 8) * 8 + 1 = 185` frames, dimension 1024 stays 1024.

#### Pipeline

```python
pipe = LTXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=True)
pipe.to("mps")
result = pipe(
    image=image, prompt=prompt,
    width=width, height=height,
    num_frames=num_frames,
    num_inference_steps=self.num_inference_steps,
)
frames = result.frames[0]                   # list[PIL.Image]
export_to_video(frames, str(out_path), fps=self.fps)
```

Cache key: `f"ltx::{model_id}"`. `evict_modality("video")` targets this prefix. `produces_audio = False`.

`warmup()` triggers the pipeline load (~30 s + ~10 GB download on first run). Memory peak is ~12 GB during inference; the pipeline must be the only image/video model resident.

### `fastwan_local.py` — legacy

Entire file is commented out. Was an attempt at the FastVideo / FastWan 2.2 5B model that didn't pan out (issues with the lib's i2v support at the time). Kept for archaeology; not referenced in yaml. Safe to delete.

## How `model_type` and `produces_audio` flow

Two pipeline-level decisions key off the picked video model:

1. **Per-scene clip duration** = `VideoService(...).scene_duration` (yaml `scene_duration` field). Drives both the audio fit target and the `_generate_single_video(scene["duration"]=...)` call.
2. **Skip lipsync** = `worker.produces_audio()` (yaml `produces_audio` field). When VEO 3 is active, the pipeline doesn't call `worker.apply_lipsync` even if the user enabled the lipsync toggle in the sidebar.

Both read off the same yaml entry — change one model, the whole pipeline adapts.

## Why every video backend takes a seed image

We could expose text-to-video too (Sora, VEO, LTX all support it). We don't because:

- **Iteration speed** — text-to-image is faster than text-to-video. Generating a still first lets the user pick which scenes look right before committing to expensive video gen.
- **Style consistency** — image gen with response-id chaining (OpenAI) or shared LoRAs gives cross-scene visual continuity. Text-to-video tends to drift.
- **Editability** — the user can replace any scene's image (custom upload) before video gen.

If text-to-video becomes desired later, add a separate backend method (e.g. `generate_video_text(...)`) and a different protocol — keeping the seed-image path canonical.

## Async at the worker

`POST /videos/generate` is **async** (`202 → job_id`). Every video gen is multi-minute regardless of backend, so polling is the right fit.

`WorkerClient.generate_video` polls every 5 s (the global default). Pipeline callers don't see job ids — they just `await` the `MediaAsset` return.

## Memory & eviction

Local video uses cache prefix `ltx::`. `evict_modality("video")` drops it. The cinematic pipeline calls this at the end of `generate_video` so the next phase (lipsync / merge) has headroom. Today's "next phase" is Streamlit-side ffmpeg which doesn't need GPU memory, but the eviction is still correct hygiene for future workflows that chain another GPU step.

## Failure modes

| Failure | What's raised |
| --- | --- |
| `torch` / `diffusers` missing | `BackendUnavailable("Local video backend needs torch + diffusers...")` |
| LTX dimension violates 32-divisible | Caught by `_parse_dimension`; rounds down to a valid dim. No error. |
| LTX frame count violates 8k+1 | Caught by `_round_frames`. No error. |
| Sora job returns `failed` status | `GenerationFailed("Sora Video generation failed: <id>; Reason: ...")` |
| VEO returns no videos | `GenerationFailed("VEO returned no video")` |
| Replicate URL download fails | `GenerationFailed(...)` from `core.replicate_client` |
| MPS OOM | bubbles as runtime exception → wrapped as `GenerationFailed` |

## Tuning

Most tuning is per-yaml-entry. For LTX:

- `num_inference_steps: 30` is the diffusers default; bump to 50 for more detail at 1.5× cost.
- `fps: 24` standard cinematic. Bump to 30 if you want smoother motion (proportionally more frames).
- `ram_gb: 12` should reflect actual peak — the LRU's eviction decisions depend on accuracy.

For Replicate:

- `replicate_id` is the only thing that matters; the input keys are a stable superset that works for Wan, Hunyuan, and LTX-on-Replicate.

## Future improvements

- **Real LTX text-to-video path** — `LTXPipeline` (no image conditioning). Add as `services/video/backends/ltx_t2v_local.py` if/when text-only becomes desired.
- **Sora prompt enhancement** — Sora's API has a built-in prompt enhancement option we don't expose.
- **VEO 3 Fast** — separate yaml entry for the faster (cheaper) variant once it's API-stable.
- **Per-scene seed selection** — currently uses the storyboard image. Could allow user to upload a different seed per scene.
- **Resolution upscaling** — generate at 768×768, upscale to 1024 with a separate pass. Halves cost on local LTX.
- **Frame interpolation** — RIFE / FILM post-pass to double fps without doubling generation cost.
- **Prefix `produces_video_with_text` flag** — for backends that can render baked-in subtitles. Currently the merge step does it Streamlit-side (or doesn't).
- **Delete `fastwan_local.py`** — it's been dead code for months. Either resuscitate or remove.
