---
name: 🐛 Bug report (markdown)
about: Something isn't working — broken pipeline, wrong output, crash, OOM, etc.
title: "[Bug]: "
labels: ["bug", "triage"]
assignees: ""
---

<!--
Before submitting:
- Make sure you're on the latest `main` (`git pull` + `pip install -r requirements.txt`).
- Search existing issues to avoid duplicates.
- For security issues, do NOT file here — see SECURITY.md.
-->

## Pre-flight checks

- [ ] I'm running the latest commit on `main`.
- [ ] I searched existing issues and didn't find a duplicate.
- [ ] This is not a security vulnerability.

## Summary

<!-- One or two sentences describing the bug. -->

## Steps to reproduce

1.
2.
3.

## Expected behaviour

<!-- What should have happened? -->

## Actual behaviour

<!-- What actually happened? Paste error messages verbatim. -->

```
<paste error / traceback here>
```

## Affected modality

<!-- Tick all that apply. -->

- [ ] script (LLM)
- [ ] image
- [ ] video
- [ ] lipsync
- [ ] tts
- [ ] merge / ffmpeg
- [ ] UI (Streamlit)
- [ ] worker / FastAPI
- [ ] tools (Merge / Watermark / Trim / YouTube)
- [ ] other / not sure

## Backend tier in use

<!-- Tick one. -->

- [ ] Auto
- [ ] Local (OSS, on-device)
- [ ] Cloud OSS (Replicate)
- [ ] API (OpenAI / Gemini / Sync.so / ElevenLabs / …)
- [ ] Mix (specify below)

## Specific backend / model

<!-- Exact model name from `configs/models.yaml`, e.g. `mlx-qwen-2.5-7b-q4`, `ltx-video`, `gemini-2.5-flash-image`. -->

## Environment

- OS:
- Python: <!-- python --version -->
- Imagina AI commit: <!-- git rev-parse --short HEAD -->
- ffmpeg: <!-- ffmpeg -version | head -n1 -->
- GPU / accelerator: <!-- e.g. Apple M2 16GB, NVIDIA RTX 3090, CPU only -->
- Install mode: <!-- honcho / two-terminal / docker / other -->

## Relevant logs

<!--
Paste output from `logs/` and/or the worker console.
REDACT API keys, tokens, and personal data before pasting.
-->

```
<logs here>
```

## Screenshots / sample output

<!-- Drag-and-drop images or short clips if the bug is visible. -->

## Additional context

<!-- Anything else — recent changes, workarounds you tried, related issues. -->
