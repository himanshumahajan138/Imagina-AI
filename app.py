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

from ui import sidebar, theme
from ui.tabs import cinematic, merge, trimmer, watermark, youtube

load_dotenv()


def _bootstrap_session_flags() -> None:
    for key in ["rerun_needed", "scene_videos_generated", "generating", "action"]:
        if key not in st.session_state:
            st.session_state[key] = False


def main() -> None:
    st.set_page_config(page_title="🎬 Imagina AI Video Generator", layout="wide")

    _bootstrap_session_flags()

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
