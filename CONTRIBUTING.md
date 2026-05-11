# Contributing to Imagina AI

Thanks for your interest in improving Imagina AI! This guide covers how to get a working dev environment, the conventions the project follows, and how to submit changes.

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Project layout](#project-layout)
- [Adding a new backend](#adding-a-new-backend)
- [Coding conventions](#coding-conventions)
- [Testing](#testing)
- [Commit and PR guidelines](#commit-and-pr-guidelines)
- [Reporting bugs and requesting features](#reporting-bugs-and-requesting-features)
- [Security issues](#security-issues)

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to uphold it. Please report unacceptable behaviour to **<himanshumahajan138@gmail.com>**.

## Ways to contribute

- **Bug fixes** — pick anything from the [issue tracker](https://github.com/himanshumahajan138/Imagina-AI/issues) labelled `bug` or `good first issue`.
- **New backends** — add a Local, Cloud OSS, or API tier for any modality (script, image, video, lip-sync, TTS). See [Adding a new backend](#adding-a-new-backend).
- **UI / UX** — improve the Streamlit sidebar, error messages, progress indicators, or the bundled tools (Merge, Watermark Remove, Media Trim, YouTube Download).
- **Documentation** — README, docstrings, examples, model cards in `docs/`.
- **Performance** — memory-eviction tuning, batching, caching wins on M2 / 16 GB.

If you plan a non-trivial change, please open an issue first so we can agree on the approach before you spend time on it.

## Development setup

Prerequisites: Python 3.10+, `ffmpeg` on `PATH`, and ~10 GB free disk if you want local models.

```bash
git clone https://github.com/himanshumahajan138/Imagina-AI.git
cd Imagina-AI

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # fill in at least one provider key, or run local-only
```

Run the worker and UI together with honcho:

```bash
honcho start                # worker on :8005, Streamlit on :8004
```

Or in two terminals:

```bash
uvicorn worker.main:app --port 8005 --reload
streamlit run app.py --server.port 8004
```

For local-only (no provider keys needed):

```bash
pip install -U torch diffusers transformers accelerate safetensors imageio imageio-ffmpeg
pip install mlx-lm                # Apple Silicon LLM
pip install onnxruntime-coreml    # Wav2Lip lipsync (also needs models/wav2lip.onnx)
```

Open <http://localhost:8004>. The sidebar warns if the worker isn't reachable.

## Project layout

```
app.py              # Streamlit entry point
configs/            # models.yaml — single source of truth for backend tiers
core/               # registry, tier picker, prompt builders, shared utilities
pipelines/          # end-to-end flows (script → image → video → tts → merge)
server/             # FastAPI shim used by app.py
services/           # ffmpeg, YouTube, watermark, trim helpers
worker/             # FastAPI daemon holding model weights across reruns
ui/                 # Streamlit tabs and components
tests/              # pytest suite
docs/               # model cards, design notes
```

`configs/models.yaml` is the source of truth — every backend the system can use is declared there with tier, env requirements, and per-model metadata. The registry honours `st.session_state.preferred_models[modality]` from the sidebar.

## Adding a new backend

1. **Pick a modality:** `script`, `image`, `video`, `lipsync`, or `tts`.
2. **Add a config entry** in `configs/models.yaml` under the right modality, with:
   - `tier` — one of `local`, `cloud-oss`, `api`
   - `env` — required environment variables (the registry uses these to decide availability)
   - per-model metadata the UI reads (e.g. `supported_dimensions`, `min_duration`, `max_duration`, `step`)
3. **Implement the backend** under the modality's folder (e.g. `services/image/<name>.py`) following the existing interface. Heavy models should load inside the worker (`worker/`) so they survive Streamlit reruns.
4. **Register it** so the registry can resolve it from the YAML entry.
5. **Add an eviction hook** if it holds GPU/MPS/MLX memory — pipelines call `worker.evict_models(modality=...)` at phase boundaries; new backends must release cleanly.
6. **Add at least one test** under `tests/` covering the happy path with a small fixture or mocked client.
7. **Update the README** model table.

If your backend is stub-only (no creds yet), still wire it up — the registry flags stubs and ones whose env isn't set, but keeps them user-selectable.

## Coding conventions

- **Python 3.10+**, type hints encouraged on public functions.
- **Format**: keep lines reasonable, prefer readability over cleverness.
- **No silent fallbacks** in backend code — if a provider env var is missing, surface it through the registry so the UI can warn.
- **Logging**: use the existing logger; logs land in `logs/`.
- **Secrets**: never commit `.env` or API keys; `.env.example` is the only source-controlled env file.
- **Don't add features the issue/PR doesn't require.** Keep PRs focused.

## Testing

```bash
pytest                       # full suite
pytest tests/path/to/file.py # one file
pytest -k "name_filter"      # one test
```

A PR that touches a backend should add or update tests for it. Heavy network calls should be mocked. Tests that require provider credentials must be skipped automatically when the env var isn't set.

## Commit and PR guidelines

**Commits**

- One logical change per commit.
- Imperative mood, short subject line (≤ 72 chars), longer body explaining *why* if non-obvious.
- Reference issues with `Fixes #123` or `Refs #123`.

**Pull requests**

1. Fork the repo and branch from `main`: `git checkout -b feat/my-thing`.
2. Make your changes, add tests, update docs.
3. Run the test suite locally — make sure it passes.
4. Push and open a PR against `main`. In the description include:
   - what changed and why
   - screenshots / a short clip if it affects the UI
   - test plan: what you ran, what you verified
5. Be ready to iterate — review comments are normal and welcome.

Small, focused PRs are merged much faster than large grab-bag ones. If you find unrelated cleanups while working, please put them in a separate PR.

## Reporting bugs and requesting features

Use the GitHub issue templates — they ask for the info we actually need (repro steps, env, what tier/backend you were using, logs). For feature requests, describe the user-facing outcome first, then the implementation idea (optional).

## Security issues

**Do not file a public issue** for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the disclosure process.

---

Thanks for contributing! 🎬
