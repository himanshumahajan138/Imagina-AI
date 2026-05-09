# Imagina AI — Developer Documentation

This directory is the long-form companion to the top-level [README](../README.md). It targets contributors who want to understand the *why* behind decisions, not just the *what*.

Everything here is in-tree and version-controlled with the code. When you change a subsystem materially, update its doc in the same PR.

## Map

### Foundations

- [architecture.md](architecture.md) — Big picture: process boundaries, data flow, and the design decisions that shape every other doc.
- [core.md](core.md) — Reference for the `core/` package (types, protocols, registry, errors, logger, model_manager, replicate_client, worker_client, worker_config, job_queue, storage, utils).
- [registry-and-models.md](registry-and-models.md) — `configs/models.yaml` schema, tier-picker logic, env-satisfaction rules, dimension intersection, the `unavailable` flag.
- [memory-management.md](memory-management.md) — `ModelManager`, GPU cache draining, phase-aware eviction, startup warmup, `IMAGINA_PRELOAD_LLM`.

### Worker daemon

- [worker.md](worker.md) — FastAPI daemon: routes, async job manager, error mapping, lifecycle, env vars.

### Services (one per modality)

- [services/README.md](services/README.md) — Facade pattern, backend protocols, dispatch.
- [services/llm.md](services/llm.md) — Script generation (OpenAI / Gemini / Replicate / MLX local).
- [services/image.md](services/image.md) — Cinematic stills (OpenAI / Gemini / Replicate / SDXL-Turbo / Z-Image-Turbo).
- [services/video.md](services/video.md) — Per-scene clips (Sora / VEO 3 / Replicate / LTX local).
- [services/tts.md](services/tts.md) — Voice synthesis (Kokoro / F5-TTS / ElevenLabs).
- [services/lipsync.md](services/lipsync.md) — Talking-head sync (Sync.so / LatentSync / Wav2Lip local).
- [services/media.md](services/media.md) — ffmpeg helpers: merge, trim, watermark, logo, YouTube downloader.

### Pipelines & UI

- [pipelines/README.md](pipelines/README.md) — Orchestration layer overview, why pipelines are separate from services.
- [pipelines/cinematic.md](pipelines/cinematic.md) — Theme → finished short film: phases, file handoff, error recovery.
- [pipelines/srt.md](pipelines/srt.md) — SRT-style script loading shim.
- [audio-pipeline.md](audio-pipeline.md) — TTS speed pre-estimation + duration fit + BGM mixing + lipsync handoff.
- [ui.md](ui.md) — Streamlit layout: sidebar, tabs, components, session state, tier picker.

### Working on the codebase

- [extending.md](extending.md) — How to add a new backend, modality, or UI tab without touching shared infrastructure.
- [roadmap.md](roadmap.md) — Known limitations, deferred work, places worth improving.

### Archived

- [_archive/](_archive/) — Pre-implementation plans kept for historical context. The corresponding subsystems are now built; refer to the live docs above.

## Conventions

- **File:line refs** are clickable in most editors and on GitHub: `core/registry.py:75` jumps to `pick_model`.
- **"Why"** sections explain non-obvious choices. If a piece of code surprises you, look for the why before "fixing" it.
- **"Future improvements"** sections are dumping grounds for known-better-but-not-now ideas. Promote them to issues when you pick one up.
- Code blocks marked `txt` are diagrams; `python`, `yaml`, `bash` are runnable.
- Cross-references use relative links: `[memory-management.md](memory-management.md)`.

## Status

These docs describe the codebase as of the current commit. They were written by walking the source — if you find a mismatch, the source is canonical and the doc is stale. PRs welcome.
