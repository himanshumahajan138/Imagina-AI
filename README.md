# Imagina AI

![Imagina AI logo (light)](images/logo-white.png)

A full-stack Streamlit studio for generating cinematic videos from themes or structured scripts, with optional custom audio + lip-sync, FastWan/Sora/Gemini backends, watermark removal, media trimming, and YouTube downloading.

## Table of Contents

- Features
- Architecture at a Glance
- Requirements
- Installation
- Configuration (.env)
- Running the App
- Usage Guide
- Voice & Model Notes
- Troubleshooting
- Project Layout
- Open Source Guidelines & Contributing
- Author
- License

## Features

- **Cinematic generator**: Create scripts from a theme or upload structured scripts; generate scene images/videos via OpenAI Sora, Gemini/VEO, or FastWan; merge scenes into a final render.
- **Audio pipeline**: Built-in VibeVoice TTS voices, optional custom BGM, and optional Sync.so lip-sync for custom audio.
- **Editing tools**: Merge videos, remove watermarks (OpenCV inpainting or FFmpeg delogo), trim media, and download from YouTube (audio/video/both) via `yt-dlp`.
- **Output controls**: Aspect ratios, resolutions, watermark/logo toggles, reference images, per-scene regeneration, and custom speaker/pitch/speed.
- **Local assets**: Logos in `images/`, sample script in `samples/sample.txt`, bundled streaming TTS voices in `TTS/voices/streaming_model`.

## Architecture at a Glance

- **UI**: `app.py` Streamlit app with tabs for generation, merging, watermark removal, trimming, and YouTube downloads.
- **Generation logic**: `core/utils.py` handles script/image/video/audio generation, merging, and lip-sync integration.
- **Models/config**: `core/config.py` defines model types, dimensions, and resolutions; `core/fastwan_utils.py` wraps FastWan.
- **Media ops**: `core/trimmer_utils.py` for trim/preview; `core/yt_downloader_utils.py` for YouTube via `yt-dlp` + `ffmpeg`.
- **Static server**: `core/static_file_serve_api.py` FastAPI server (optional ngrok) for hosting temp files used by lip-sync.
- **TTS**: `TTS/audio_generation_pipeline.py` and `TTS/voices` for VibeVoice streaming voices.

## Requirements

- Python 3.10+ (Torch/diffusers friendly)
- ffmpeg CLI on PATH
- yt-dlp on PATH (for YouTube tab)
- (GPU recommended) CUDA-capable torch build for heavy gen tasks

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
# Optional: pins in custom_req.txt if you need stricter/GPUs-specific versions
```

## Configuration (.env)

Create `.env` in the repo root:

```env
OPENAI_API_KEY=sk-...           # Sora video + GPT prompts
GOOGLE_GENAI_API_KEY=...        # Gemini/VEO workflows
SYNC_API_KEY=...                # Lip-sync via syncsdk

BASE_URL=http://localhost:8000  # Static server base for temp uploads
STATIC_SERVER_PORT=8000         # Port for static server
USE_NGROK=true                  # Set false to skip ngrok
NGROK_AUTH_TOKEN=...            # Optional, if using ngrok

FASTWAN_RUNPOD_API_KEY=...      # Optional: if using FastWan via RunPod endpoints
ENDPOINT_ID=...                 # Optional: FastWan endpoint id
WATERMARK_PATH=images/watermark.png   # Override if using a custom watermark
```

Place logos/watermarks in `images/` or point paths accordingly.

## Running the App

```bash
# (optional) serve temp files for lip-sync/static access
uvicorn core.static_file_serve_api:app --host 0.0.0.0 --port 8000

# launch the Streamlit UI
streamlit run app.py --server.address 0.0.0.0 --server.port 8004 --server.enableCORS false
```

## Usage Guide

- **Cinematic Generator**
  1) Pick model type (SORA/OpenAI, VEO/Gemini, FastWan), language, dimensions, resolution, duration.
  2) Generate a script from a theme or upload a script file (see "Script format" helper in UI).
  3) Optionally upload reference images, choose speaker/voice, pitch/speed, and custom BGM; toggle watermark/logo.
  4) Click "Generate Scenes (and Audio)" to create per-scene assets; review in the gallery and regenerate scenes as needed.
  5) Click "Generate Final Video" to merge scenes; enable lip-sync if using custom audio.
  6) Download the final render in the UI.
- **Merge Videos**: Combine clips with ffmpeg concat.
- **Watermark Remover**: Select region, choose OpenCV inpainting (quality) or FFmpeg delogo (speed), download cleaned video.
- **Media Trimmer**: Select time range, preview, trim, and download.
- **YouTube Downloader**: Fetch audio, video, hybrid, or both via `yt-dlp`.

## Voice & Model Notes

- Streaming voices bundled in `TTS/voices/streaming_model`; `TTS/download_experimental_voices.sh` can fetch extras.
- FastWan sizes come from `core/config.py::FASTWAN_DIMENSIONS`; Sora size mappings in `SORA_DIMENSIONS`.
- Lip-sync needs reachable URLs; run the static server and optionally ngrok (`USE_NGROK=true`).

## Troubleshooting

- **ffmpeg/yt-dlp not found**: Install and ensure they are on PATH.
- **GPU performance**: Use a CUDA torch build; large generations need GPU + VRAM.
- **Lip-sync fails**: Check `BASE_URL`/ngrok reachability and `SYNC_API_KEY`.
- **YouTube download errors**: Ensure `yt-dlp` (not just `youtube-dl`) is installed and updated.
- **Disk space**: Temp files are written to your system temp dir; large runs can consume space.

## Project Layout

- `app.py` - Streamlit UI and workflows.
- `core/utils.py` - script/image/video/audio generation, merging, lip-sync wiring.
- `core/config.py` - model/dimension/resolution options.
- `core/fastwan_utils.py` - FastWan video generation helper.
- `core/trimmer_utils.py` - trimming helpers and ffmpeg wrappers.
- `core/yt_downloader_utils.py` - YouTube downloads via `yt-dlp` + `ffmpeg`.
- `core/static_file_serve_api.py` - FastAPI static server with optional ngrok.
- `TTS/` - VibeVoice TTS pipeline, voices, demos/configs.
- `images/` - logos and watermark; `samples/` - script format example.

## Open Source Guidelines & Contributing

- See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow, testing, and pull request expectations.
- File issues for bugs/features with clear repro steps, logs, and environment info when possible.

## Author

- [Himanshu Mahajan](https://www.github.com/himanshumahajan138)

## License

MIT — see [LICENSE](LICENSE).

Made with ❤️ by [Himanshu Mahajan](https://www.github.com/himanshumahajan138)
