# Media Service — ffmpeg / OpenCV Helpers

Pure utility module — no `service.py` facade, no model calls. These are the primitives the cinematic pipeline and UI tabs use to manipulate audio and video files.

Most run **Streamlit-side**, but the lipsync chunk-and-stitch path calls them server-side (via `services/lipsync/backends/sync_api.py`), so they're written headless: log via `logger`, no `st.*` calls in functions reachable from the worker.

## Files

```
services/media/
  __init__.py
  trimmer.py        ffprobe duration + codec; ffmpeg trim; Streamlit display widgets (UI-only)
  merger.py         ffmpeg concat-demuxer; chunked split for long media
  watermark.py      OpenCV inpaint + ffmpeg delogo; watermark/logo overlay; centre-crop
  youtube.py        yt-dlp wrapper for the YouTube Downloader tab
```

## `trimmer.py`

Two halves separated by a comment divider:

### Headless helpers (server-safe)

```python
def get_file_duration(file_path: str) -> float:
    """ffprobe -show_entries format=duration → float seconds, or 0 on error."""

def format_time(seconds: float) -> str:
    """seconds → 'HH:MM:SS'."""

def get_codec_info(file_path: str) -> list:
    """ffprobe -show_entries stream=codec_type,codec_name → list of stream dicts."""

def trim_media(file_path, start_time, end_time, output_path,
               progress_placeholder=None) -> bool:
    """ffmpeg -ss/-to copy. progress_placeholder is optional Streamlit st.empty()."""
```

`trim_media`'s `progress_placeholder` parameter is the only Streamlit knob — passing `None` (the worker does) just skips the status update. On error the function logs via `core.logger` and returns `False`; callers decide whether to surface the failure.

**Why `progress_placeholder` is optional and not an interface**: the trimmer is reachable from the worker via `merger.split_media_into_chunks → trim_media`, where there's no Streamlit context. Keeping the parameter optional avoids splitting the function into "headless" and "with-UI" variants.

### Streamlit display widgets (UI-only)

```python
def display_timeline(duration, start, end):           # 4-column metrics row
def display_file_info(file_path, file_name):          # 2-column file metadata
def is_audio_file(file_name) -> bool                  # extension check
def display_media_player(file_path, file_name):       # st.audio or st.video
```

Used by `ui/tabs/trimmer.py`. Not reachable server-side.

`import streamlit as st` at the top of the file is fine for the UI-only helpers — importing streamlit doesn't require a script context, only *calling* `st.*` functions does.

## `merger.py`

Headless. Two functions:

```python
def split_media_into_chunks(file_path: str, max_duration: float = 299) -> list[(start, end, chunk_path)]:
    """Slice a long file into <= max_duration chunks via ffmpeg copy.
    Total chunks = ceil(total_duration / max_duration)."""

def merge_videos(video_chunks: list[str], output_path: str) -> bool:
    """ffmpeg concat demuxer. Errors logged, returns success bool."""
```

Used by `services/lipsync/backends/sync_api.py` for Sync.so's 5-minute input limit. Could be useful elsewhere — concat is a generic operation.

`merge_videos` writes a temp `concat_list.txt` with `file 'path'` lines, runs:

```
ffmpeg -f concat -safe 0 -i concat_list.txt -c copy -y out.mp4
```

`-c copy` means stream copy — no re-encoding, fast. Requires inputs with matching codecs. For arbitrary input, you'd need `-c:v libx264 -c:a aac` instead.

## `watermark.py`

Mixed bag of video post-processing helpers:

### Watermark removal (UI-only paths)

Used by the "Video Watermark Remover" tab. Two methods:

```python
def remove_watermark_opencv(input_path, output_path, x, y, width, height, method="telea"):
    """OpenCV inpaint per frame, audio passthrough via ffmpeg."""

def remove_watermark_ffmpeg(input_path, output_path, x, y, width, height):
    """ffmpeg delogo filter — faster, blurrier."""
```

The OpenCV path is slower (per-frame Python loop) but produces cleaner results on textured backgrounds. The ffmpeg `delogo` filter is fast but obvious on plain backgrounds.

### Watermark / logo overlay (cinematic pipeline)

```python
def watermark_addition(final_output) -> Path:
    """Overlay images/watermark.png centred on the video. Re-encodes."""

def logo_addition(video_path, logo_path, position="top-right") -> Path:
    """Overlay a user logo at top-left or top-right with -1:30 scale."""
```

Both run Streamlit-side in `pipelines/cinematic.py:_generate_single_video` (per-scene) or `final_generation` (final pass). The user toggles them in the sidebar.

The watermark image is a project asset at [images/watermark.png](../../images/watermark.png). The overlay uses `-vf overlay=...` with `colorchannelmixer aa=0.9` for slight transparency.

`logger.error` + `st.warning` on failure — these are Streamlit-side (not reachable from the worker), so the `st.*` call is fine.

### Helpers for other backends

```python
def crop_image_to_dimension(image_path, target_dimension) -> str:
    """Centre-crop + resize a still to match target_dimension ('1280x720'). Returns new path."""

def normalize_veo3_video(input_path) -> Path:
    """Re-encode VEO output for predictable downstream ffmpeg processing.
    libx264 + aac + faststart."""

def extract_frame(video_path, frame_number=0):
    """Read a single frame via cv2.VideoCapture for preview."""
```

`crop_image_to_dimension` is called by `services/video/backends/openai.py` (Sora) to fit the seed image to Sora's accepted dimensions before upload.

`normalize_veo3_video` is called by `watermark_addition` and `logo_addition` to reencode VEO 3 output before overlay — VEO uses non-standard moov/mdat ordering that breaks naïve overlay pipelines.

## `youtube.py`

Used only by the YouTube Downloader tab in the UI. Wraps `yt-dlp` for direct downloads + audio extraction. Not part of the cinematic pipeline.

## How these fit into the cinematic flow

```txt
generate_video (Streamlit-side)
    ├── _generate_single_video (per scene)
    │   ├── worker.generate_video → mp4 at temp_file
    │   ├── if use_logo: services.media.watermark.logo_addition
    │   └── if watermark: services.media.watermark.watermark_addition
    │
final_generation (Streamlit-side)
    ├── ffmpeg.concat over per-scene mp4s → merged_tmp
    ├── if lipsync_mode and not produces_audio:
    │     worker.apply_lipsync (worker-side) → calls services.lipsync.backends.sync_api
    │       ├── split_media_into_chunks (headless, server-side)
    │       ├── upload_file_to_static_server (server-side)
    │       ├── per-chunk Sync.so call
    │       └── merge_videos (headless, server-side)
    └── ffmpeg encode + audio mux + scale → final_output
```

So `merger.py` and the headless `trimmer.py` helpers run **server-side** through the lipsync flow. `watermark.py` runs **Streamlit-side** for cinematic post-processing and as the watermark-removal tool. Both are correct.

## Why this isn't a `MediaService` facade

The other modalities have provider-swappable backends (`OpenAI` vs `Replicate` vs `local`). ffmpeg is the one tool, no swappability needed. Wrapping it in a service class would be ceremony for nothing.

If some day we want to swap ffmpeg for, say, `moviepy` or remote-encoding-as-a-service, that's the moment to introduce a facade. Not before.

## Failure modes

| Failure | What happens |
| --- | --- |
| ffmpeg / ffprobe binary missing | subprocess raises `FileNotFoundError`; functions log and return `0` / `False` |
| Input file unreadable | ffprobe / ffmpeg returns non-zero; functions log stderr and return failure |
| OpenCV `cv2.VideoCapture` returns no frames | functions return early; caller sees empty result |
| Concat demuxer rejects mismatched codecs | ffmpeg returns non-zero; `merge_videos` logs stderr |
| Watermark image not found at `images/watermark.png` | `watermark_addition` is skipped (gated by `Path(WATERMARK).exists()` check in pipeline) |

## Future improvements

- **`MediaService` facade** if/when we want remote encoding (LambdaCloud, Cloudflare Stream).
- **Hardware encoding** — `-c:v h264_videotoolbox` on M2 is much faster than libx264. Currently we use software encode for portability; an env-driven preference would help.
- **Proper progress reporting** — `trim_media`'s `progress_placeholder` is binary (working/done). Parsing ffmpeg `-progress pipe:1` would give frame-level progress.
- **Concat without re-encode** when codecs already match — currently `merge_videos` already does `-c copy`; per-scene videos with VEO output may not match → enforces re-encode upstream.
- **Move the static file server** out of `server/` and into the worker as `POST /upload`. Eliminates one moving piece in the stack.
- **Split `watermark.py`** — it's three different things (watermark removal, watermark overlay, image processing helpers) crammed in one file. Splitting along those axes would make imports more explicit.
