"""Theme + branding helpers (logo path resolution)."""

from __future__ import annotations

from streamlit_theme import st_theme


def resolve_logo_path() -> str:
    """Return the right logo image for the current Streamlit theme."""
    streamlit_theme = st_theme(key="streamlit_theme")
    if streamlit_theme and streamlit_theme.get("base") == "light":
        return "images/logo-white.png"
    return "images/logo-black.png"
