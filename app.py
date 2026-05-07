"""Imagina AI — Streamlit entry point.

This file is intentionally tiny: it composes the global page config,
sidebar, and the five tabs. All actual logic lives under `services/`,
`pipelines/`, and `ui/`.

Run:
    streamlit run app.py --server.address 0.0.0.0 --server.port 8004
"""

from __future__ import annotations

from dotenv import load_dotenv

import streamlit as st

from core.worker_client import worker
from core.worker_config import WORKER_URL
from ui import sidebar, theme
from ui.tabs import cinematic, merge, trimmer, watermark, youtube

load_dotenv()


def _bootstrap_session_flags() -> None:
    for key in ["rerun_needed", "scene_videos_generated", "generating", "action"]:
        if key not in st.session_state:
            st.session_state[key] = False


def _check_worker() -> None:
    """Banner if the worker daemon is unreachable.

    Cached for the session so we don't ping every rerun. User can clear
    `worker_alive` from session_state to re-check after starting the worker.
    """
    if "worker_alive" not in st.session_state:
        st.session_state.worker_alive = worker.is_alive()
    if not st.session_state.worker_alive:
        st.error(
            f"⚠️ Imagina worker is not reachable at `{WORKER_URL}`. "
            "Generation will fail until it's started.\n\n"
            "Start it with `honcho start` (recommended) or "
            "`uvicorn worker.main:app --port 8005` in another terminal, "
            "then refresh this page."
        )


def main() -> None:
    st.set_page_config(page_title="🎬 Imagina AI Video Generator", layout="wide")

    _bootstrap_session_flags()
    _check_worker()

    logo_path = theme.resolve_logo_path()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🎬 Cinematic Generator",
            "➕ Merge Videos",
            "✂️ Video Watermark Remover",
            "✂️ Media Trimmer",
            "📽️ YouTube Downloader",
        ]
    )

    sidebar.render(logo_path)

    with tab1:
        cinematic.render()
    with tab2:
        merge.render()
    with tab3:
        watermark.render()
    with tab4:
        trimmer.render()
    with tab5:
        youtube.render()


main()
