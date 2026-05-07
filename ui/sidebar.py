"""Global settings sidebar.

Renders the model/voice/dimension/duration/etc. controls and the script
input expanders. All state is stored on `st.session_state` for tabs to read.
"""

from __future__ import annotations

import streamlit as st

from core.config import (
    COMMON_LANGUAGES,
    DIMENSIONS,
    RESOLUTIONS,
    SPEAKER_OPTIONS,
)
from core.registry import common_dimensions
from services.video.service import video_constraints
from ui.components import tier_picker


def _filtered_dimensions() -> dict[str, str]:
    """DIMENSIONS limited to what both image + video backends actually support.

    Falls back to the full list (with a warning) when the chosen image and
    video models declare disjoint supported_dimensions, so the user is
    never stuck with an empty dropdown.
    """
    allowed = set(common_dimensions(["image", "video"]))
    filtered = {label: val for label, val in DIMENSIONS.items() if val in allowed}
    if not filtered:
        st.warning(
            "⚠️ Selected image and video models share no compatible dimensions; "
            "showing all options. Generation may resize to fit."
        )
        return DIMENSIONS
    return filtered


def render(logo_path: str) -> None:
    with st.sidebar:
        st.sidebar.image(logo_path)
        st.header("🎛️ Global Settings", divider="red")

        tier_picker.render()

        # Mirror the chosen video model into legacy `model_type` so any
        # callers still reading it (image refinement default, watermark
        # framing helpers) keep working until they're fully migrated.
        video_model = st.session_state.preferred_models.get("video")
        if video_model == "veo-3":
            st.session_state.model_type = "gemini"
        elif video_model == "sora":
            st.session_state.model_type = "openai"
        else:
            st.session_state.model_type = "local"

        constraints = video_constraints()

        with st.expander("🎤 Voice & Video Settings", expanded=True):
            st.session_state.language = st.selectbox(
                "Language",
                COMMON_LANGUAGES,
                index=(
                    list(COMMON_LANGUAGES.keys()).index("American English")
                    if "American English" in COMMON_LANGUAGES
                    else 0
                ),
                disabled=st.session_state.generating,
            )
            dimension_options = _filtered_dimensions()
            st.session_state.dimension = st.selectbox(
                "Video Dimensions",
                dimension_options.keys(),
                disabled=st.session_state.generating,
            )
            st.session_state.download_quality = st.selectbox(
                "Download Resolution",
                RESOLUTIONS.keys(),
                index=1,
                disabled=st.session_state.generating,
            )

            st.session_state.duration = st.slider(
                "Duration (seconds)",
                constraints["min"],
                constraints["max"],
                constraints["default"],
                step=constraints["step"],
                disabled=st.session_state.generating,
            )
            st.session_state.watermark = st.toggle(
                "Watermark",
                True,
                key="watermark_toggler",
                disabled=st.session_state.generating,
            )
            st.session_state.image_refinement_mode = st.toggle(
                "Image Refinement Mode",
                True,
                key="image_refinement_mode_toggler",
                disabled=st.session_state.generating,
            )

            if st.session_state.image_refinement_mode:
                with st.container():
                    custom_reference_imgs = st.file_uploader(
                        "Upload Custom Reference Images (3 MAX)",
                        type="png",
                        accept_multiple_files=True,
                        key="custom_reference_imgs",
                    )
                    if custom_reference_imgs and len(custom_reference_imgs) > 3:
                        st.warning(
                            "⚠️ You can upload up to 3 images only. "
                            f"You uploaded {len(custom_reference_imgs)}."
                        )
                        custom_reference_imgs = custom_reference_imgs[:3]

                st.session_state.custom_reference_images = custom_reference_imgs
            else:
                st.session_state.custom_reference_images = []

            st.session_state.use_custom_audio = st.toggle(
                "Use Custom Audio",
                False if st.session_state.model_type == "gemini" else True,
                key="custom_audio_toggler",
                disabled=st.session_state.generating,
            )

            if st.session_state.use_custom_audio:
                with st.container():
                    st.toggle(
                        "Lip Sync Mode",
                        value=False,
                        width="stretch",
                        disabled=st.session_state.generating,
                        key="lipsync_mode",
                    )
                    st.selectbox(
                        "Speaker",
                        list(SPEAKER_OPTIONS.keys()),
                        index=list(SPEAKER_OPTIONS.keys()).index("Heart"),
                        disabled=st.session_state.generating,
                        key="selected_speaker",
                    )
                    st.slider(
                        "Speed",
                        0.0, 2.0, 1.0, step=0.1,
                        disabled=st.session_state.generating,
                        key="selected_speed",
                    )
                    st.file_uploader(
                        "Upload Custom BGM (.wav)",
                        type="wav",
                        disabled=st.session_state.generating,
                        key="custom_bgm",
                    )
            else:
                st.session_state.selected_speaker = list(SPEAKER_OPTIONS.keys()).index("Heart")
                st.session_state.selected_speed = None

            st.session_state.use_logo = st.toggle(
                "Add Custom Logo",
                False,
                key="custom_logo_toggler",
                disabled=st.session_state.generating,
            )
            if st.session_state.use_logo:
                with st.container():
                    st.selectbox(
                        "Logo Location",
                        ["top-left", "top-right"],
                        index=1,
                        disabled=st.session_state.generating,
                        key="logo_location",
                    )
                    st.file_uploader(
                        "Upload Custom LOGO (.png)",
                        type="png",
                        disabled=st.session_state.generating,
                        key="custom_logo",
                    )

        st.divider()

        with st.expander("📝 Generate Script from Theme"):
            st.text_area(
                "Theme",
                placeholder="Enter the theme here",
                disabled=st.session_state.generating,
                key="theme",
                max_chars=1000,
            )
            st.button(
                "🎥 Generate Script",
                width="stretch",
                key="generate_script_button",
                disabled=st.session_state.generating or not st.session_state.theme.strip(),
            )

        st.divider()

        with st.expander("📁 Upload Existing Script File"):
            with st.expander("📄 Sample format for script"):
                st.markdown(
                    """
                    ### 🧾 Format Guidelines

                    To ensure accurate timing and synchronization, your script must follow a **block-based format**, where **each block contains**:

                    - A timestamp line (start and end time)
                    - One `[script]:` line
                    - One `[scene]:` line
                    - One `[video_scene]:` line

                    ---
                    #### 📌 Correct Block Structure:

                    Each block must look like this (including line breaks):

                    ```
                    00:00:00,000 --> 00:00:03,500
                    [script]: "Your dialogue or narration here."
                    [scene]: "Your visual scene description here."
                    [video_scene]: "Your camera or cinematic instruction here."
                    ```

                    ✅ Important: After every block, **leave one blank line** to separate it from the next.

                    ```
                    00:00:03,500 --> 00:00:07,000
                    [script]: "Next line of dialogue..."
                    [scene]: "Next scene description..."
                    [video_scene]: "Next video shot or framing guidance..."
                    ```

                    ---
                    ### ✅ Rules and Best Practices:

                    - ⏱️ **Timestamps are required** at the top of each block in SRT format:
                    `HH:MM:SS,mmm --> HH:MM:SS,mmm`
                    - 🏷️ Tags `[script]:`, `[scene]:`, and `[video_scene]:` are **mandatory** and **case-insensitive**.
                    - 🔄 Each `[script]` must be immediately followed by `[scene]`, then `[video_scene]` — **no skipping, no reordering**.
                    - 🗨️ Content for all tags **must be enclosed in double quotes**: `"like this"`.
                    - ↩️ Always **add a blank line between blocks**. This helps the parser distinguish them cleanly.
                    - ⚠️ If any block is missing a timestamp, script, scene, or video_scene — or uses the wrong format — it will be skipped with a warning.
                    """
                )

            st.file_uploader(
                "Upload Script (.txt)",
                type="txt",
                disabled=st.session_state.generating,
                key="script_file",
            )
            st.button(
                "📄 Load Script",
                width="stretch",
                key="load_script_button",
                disabled=st.session_state.generating or st.session_state.script_file is None,
            )
