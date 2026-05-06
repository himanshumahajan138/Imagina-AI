"""Tier / model picker for the sidebar.

Renders a per-modality selectbox of available models. Writes the user's
choice to `st.session_state.preferred_models[modality]` which the service
facades read via `core.registry.session_preferred()`.

"Auto" lets the registry decide based on env vars (api → cloud_oss → local).
"""

from __future__ import annotations

import streamlit as st

from core.registry import available_models, label_for, list_models


_MODALITIES: list[tuple[str, str]] = [
    ("llm", "📝 Script (LLM)"),
    ("image", "🖼️ Image"),
    ("video", "🎬 Video"),
    ("lipsync", "👄 Lip-sync"),
    ("tts", "🗣️ TTS"),
]

_AUTO_LABEL = "Auto (registry decides)"


def _ensure_session() -> None:
    if "preferred_models" not in st.session_state:
        st.session_state.preferred_models = {}


def render() -> None:
    """Render the tier-picker expander inside the sidebar."""
    _ensure_session()

    with st.expander("🎚️ Model Selection (advanced)", expanded=False):
        st.caption(
            "Override the auto-pick for any modality. Models without their "
            "API key / token configured are still shown but greyed by label."
        )

        for modality, header in _MODALITIES:
            available = available_models(modality)
            available_ids = {mid for mid, _ in available}
            all_ids = list(list_models(modality).keys())

            options = [_AUTO_LABEL] + all_ids
            current = st.session_state.preferred_models.get(modality)
            try:
                idx = options.index(current) if current else 0
            except ValueError:
                idx = 0

            def _format(opt: str) -> str:
                if opt == _AUTO_LABEL:
                    return _AUTO_LABEL
                base = label_for(modality, opt)
                if opt not in available_ids:
                    base += "  ·  ⚠ env not set"
                return base

            choice = st.selectbox(
                header,
                options=options,
                index=idx,
                format_func=_format,
                key=f"tier_picker_{modality}",
                disabled=st.session_state.get("generating", False),
            )

            if choice == _AUTO_LABEL:
                st.session_state.preferred_models.pop(modality, None)
            else:
                st.session_state.preferred_models[modality] = choice
