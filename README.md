# Imagina AI (refactor branch)

Three-tier generation pipeline:

- **Local OSS** — runs on M2 16 GB (Qwen 2.5 / SDXL Turbo / LTX-Video / Wav2Lip / Kokoro)
- **Cloud OSS** — Replicate / fal.ai-hosted SOTA (DeepSeek V3 / FLUX / Hunyuan / LatentSync / F5-TTS)
- **API** — OpenAI, Gemini, Sync.so, ElevenLabs

This folder is the in-progress refactor of the project at the repository
root. When the layout stabilises it will replace the root code or move
to its own repository.

## Layout

```
core/        shared infra (registry, protocols, model manager, types)
services/    one isolated module per modality (llm / image / video / lipsync / tts / media)
pipelines/   multi-service workflows (cinematic, srt)
ui/          streamlit UI, broken into tabs
server/      FastAPI static file server (for lip-sync URLs)
configs/     models.yaml registry
scripts/     doctor, download_models
```

See repo-root `README.md` for product features and usage. This README
will be promoted when the refactor is ready to take over.

## Running (during refactor)

```bash
cd imagina-ai
streamlit run app.py
```

`app.py` and `core/utils.py` here are still the monoliths from the root
(Phase 0). Phase 1 splits them into the structure above.
