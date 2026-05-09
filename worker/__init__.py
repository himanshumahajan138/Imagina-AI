"""Imagina worker daemon — FastAPI service holding all model calls.

The Streamlit app talks to this daemon over HTTP so model weights stay
loaded across reruns and long generations don't freeze the UI thread.

See `docs/worker-service.md` for design + ownership split.
"""
