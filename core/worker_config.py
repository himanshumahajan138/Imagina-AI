"""Configuration for the worker daemon URL + timeouts.

The Streamlit app and any other client read these env vars to find the
worker. Keep this small so swapping the worker out for a remote deploy
is a one-line env change.
"""

from __future__ import annotations

import os

WORKER_URL = os.getenv("IMAGINA_WORKER_URL", "http://127.0.0.1:8005")

# Sync endpoints (script/tts) — needs to cover first-time local-model
# loads, which can take 30-60s for MLX-LM Qwen 2.5 7B Q4 (~6 GB) or
# Kokoro (~1 GB). Image and video gen are async (job_id + poll) so they
# don't share this ceiling. Default chosen to comfortably absorb a
# full cold LLM load + first inference.
SYNC_TIMEOUT_S = float(os.getenv("IMAGINA_WORKER_SYNC_TIMEOUT", "600"))

# Polling cadence for async jobs.
DEFAULT_POLL_INTERVAL_S = float(os.getenv("IMAGINA_WORKER_POLL_INTERVAL", "5"))
DEFAULT_JOB_TIMEOUT_S = float(os.getenv("IMAGINA_WORKER_JOB_TIMEOUT", "1800"))
