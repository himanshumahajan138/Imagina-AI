# Contributing to Imagina AI

Thanks for helping improve Imagina AI! This guide keeps contributions consistent and easy to review.

## Quick Start
1) Fork/branch from `main`.
2) Create a virtualenv and install deps: `pip install -r requirements.txt`.
3) Ensure `ffmpeg` and `yt-dlp` are on PATH; set a `.env` with dummy keys if needed for smoke tests.

## Workflow
- Prefer small, focused PRs with a clear scope.
- Open an issue first for significant changes; describe motivation, scope, and impact.
- Use feature branches named like `feature/...` or `fix/...`.

## Coding Standards
- Python 3.10+; keep code readable and well-structured.
- Add/adjust docstrings where behavior is non-obvious; keep comments concise.
- Match existing style (Streamlit UI patterns, helpers in `core/utils.py`); avoid introducing unused deps.

## Testing & Verification
- Run targeted checks relevant to your change (e.g., unit tests if present, or manual smoke of the affected tab/workflow in Streamlit).
- For media/ffmpeg changes, include command examples or screenshots in the PR if automated tests are impractical.
- Validate that new environment variables are documented in `README.md`.

## Pull Requests
- Summarize what changed and why; call out user-visible behavior changes.
- List testing performed (commands and results); note any known gaps.
- Keep diffs minimal: remove dead code and debug prints.

## Reporting Issues
- Include reproduction steps, expected vs actual behavior, logs/tracebacks, OS/Python versions, and whether ffmpeg/yt-dlp are installed.

Thanks again for contributing!
