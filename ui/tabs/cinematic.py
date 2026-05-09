"""Cinematic Generator tab."""

from __future__ import annotations

import base64

import pandas as pd
import streamlit as st

from core.config import DIMENSIONS, RESOLUTIONS, SPEAKER_OPTIONS
from core.registry import session_preferred
from core.worker_client import worker
from pipelines.cinematic import (
    final_generation,
    generate_audio_images,
    generate_video,
    hash_df,
)
from services.llm.parser import parse_script_scene_content, validate_script_data
from ui.components.storyboard_gallery import storyboard_gallery, video_gallery


def _task_handler(action_name, task_func):
    if (
        st.session_state.get("generating")
        and st.session_state.get("action") == action_name
    ):
        task_func()
        st.session_state.generating = False
        st.session_state.action = None
        st.rerun()


def render() -> None:
    st.title(":clapper: :red[**Cinematic Theme to Video Generator**] :clapper:")

    if st.session_state.generate_script_button:
        st.session_state.generating = True
        st.session_state.action = "generate_script"
        st.rerun()

    if st.session_state.load_script_button and st.session_state.script_file:
        st.session_state.generating = True
        st.session_state.action = "load_script"
        st.session_state.uploaded_content = st.session_state.script_file.read().decode(
            "utf-8"
        )
        st.rerun()

    def _generate_script_task() -> None:
        df = worker.generate_script(
            theme=st.session_state.theme,
            duration=st.session_state.duration,
            language=st.session_state.language,
            model_id=session_preferred("llm"),
            model_type=st.session_state.model_type,
        ).assign(
            speed=st.session_state.selected_speed,
            speaker=st.session_state.selected_speaker,
            custom_image=None,
        )
        st.session_state.script_df = df
        # No eviction here. The LLM stays resident through script
        # iteration (edits, regens, multiple uploads) — eviction happens
        # at the entry to the next phase (generate_audio_images), so
        # repeated script work doesn't pay reload cost.

    _task_handler("generate_script", _generate_script_task)

    def _load_script_task() -> None:
        st.session_state.script_df = pd.DataFrame(
            parse_script_scene_content(st.session_state.uploaded_content)
        ).assign(
            speed=st.session_state.selected_speed,
            speaker=st.session_state.selected_speaker,
            custom_image=None,
        )
        st.session_state.uploaded_content = None
        # No eviction here either — same reasoning as _generate_script_task.

    _task_handler("load_script", _load_script_task)

    if "script_df" not in st.session_state:
        st.session_state.script_df = pd.DataFrame(
            columns=[
                "script",
                "scene",
                "video_scene",
                "start_time",
                "end_time",
                "speed",
                "speaker",
                "custom_image",
            ]
        )

    with st.expander("📝 Edit Script", expanded=True):
        df_display = st.session_state.script_df.copy()
        df_display.index = range(1, len(df_display) + 1)
        df_display.index.name = "Scene No."

        st.session_state.edited_df = st.data_editor(
            df_display,
            num_rows="dynamic",
            width="stretch",
            hide_index=False,
            column_config={
                "speaker": st.column_config.SelectboxColumn(
                    "Speaker",
                    options=list(SPEAKER_OPTIONS.keys()),
                    required=True,
                    disabled=False if st.session_state.use_custom_audio else True,
                ),
                "speed": st.column_config.NumberColumn(
                    "Speed",
                    min_value=0.0,
                    max_value=2.0,
                    step=0.1,
                    required=True,
                    disabled=False if st.session_state.use_custom_audio else True,
                ),
                "start_time": st.column_config.TextColumn("Start Time", required=True),
                "end_time": st.column_config.TextColumn("End Time", required=True),
                "custom_image": st.column_config.ImageColumn("Custom Image"),
            },
            disabled=st.session_state.generating,
        )

        if len(st.session_state.script_df):
            with st.expander("🎨 Upload/Delete Custom Image for a Scene"):
                st.session_state.scene_idx = st.number_input(
                    "Scene Number",
                    min_value=1,
                    max_value=len(st.session_state.script_df),
                    step=1,
                    disabled=st.session_state.generating,
                )
                with st.expander("UPLOAD"):
                    st.file_uploader(
                        "Upload Image",
                        type=["jpg", "jpeg", "png"],
                        key="custom_image_upload",
                        disabled=st.session_state.generating,
                    )
                    if st.session_state.custom_image_upload:
                        with st.expander("Preview Uploaded Image"):
                            st.image(
                                st.session_state.custom_image_upload, width="stretch"
                            )
                        if st.button(
                            f"Update Scene {st.session_state.scene_idx} with This Image",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            encoded = base64.b64encode(
                                st.session_state.custom_image_upload.read()
                            ).decode("utf-8")
                            st.session_state.script_df.at[
                                st.session_state.scene_idx - 1, "custom_image"
                            ] = f"data:image/png;base64,{encoded}"
                            st.session_state.image_updated = True
                            st.rerun()
                current_image = st.session_state.script_df.at[
                    st.session_state.scene_idx - 1, "custom_image"
                ]
                if current_image:
                    with st.expander("DELETE"):
                        st.image(current_image, width="stretch")
                        if st.button(
                            f"Delete Scene {st.session_state.scene_idx} Custom Image",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.script_df.at[
                                st.session_state.scene_idx - 1, "custom_image"
                            ] = None
                            st.session_state.image_updated = True
                            st.rerun()

    if not st.session_state.get("image_updated", False):
        current_hash = hash_df(st.session_state.edited_df.reset_index(drop=True))
        if st.session_state.get("last_df_hash") != current_hash:
            st.session_state.script_df = st.session_state.edited_df.reset_index(
                drop=True
            )
            st.session_state.last_df_hash = current_hash
            st.rerun()

    if st.button(
        (
            "🎬 Generate Scenes and Audio"
            if st.session_state.use_custom_audio
            else "🎬 Generate Scenes"
        ),
        width="stretch",
        disabled=st.session_state.generating,
    ):
        st.session_state.generating = True
        st.session_state.action = "generate_audio_image"
        st.rerun()

    def generate_audio_image_task():
        script_data = st.session_state.edited_df.to_dict(orient="records")
        errors = validate_script_data(script_data, st.session_state.duration)
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            st.session_state.generating = False
            st.session_state.action = None
            st.stop()
        generate_audio_images(
            global_dimension=DIMENSIONS[st.session_state.dimension],
            script=script_data,
            model_type=st.session_state.model_type,
            use_custom_audio=st.session_state.use_custom_audio,
        )

    _task_handler("generate_audio_image", generate_audio_image_task)

    if "video_data" in st.session_state and not st.session_state.get(
        "video_generated", False
    ):
        script_data = st.session_state.edited_df.to_dict(orient="records")
        errors = validate_script_data(script_data, st.session_state.duration)
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            st.session_state.generating = False
            st.session_state.action = None
            st.stop()

        storyboard_gallery(
            global_dimension=DIMENSIONS[st.session_state.dimension],
            script=script_data,
            model_type=st.session_state.model_type,
            use_custom_audio=st.session_state.use_custom_audio,
        )

        if st.button(
            "🎥 Generate Final Video",
            width="stretch",
            disabled=st.session_state.generating,
        ):
            st.session_state.generating = True
            st.session_state.action = "generate_final_video"
            st.rerun()

    _task_handler(
        "generate_final_video",
        lambda: st.session_state.update(
            {
                "video_generated": generate_video(
                    st.session_state.video_data,
                    st.session_state.duration,
                    DIMENSIONS[st.session_state.dimension],
                    st.session_state.model_type,
                )
            }
        ),
    )

    if st.session_state.get("scene_video_data") and st.session_state.get(
        "scene_videos"
    ):
        video_gallery(
            DIMENSIONS[st.session_state.dimension], st.session_state.model_type
        )
        if st.button(
            "📦 Final Merge",
            type="primary",
            width="stretch",
            disabled=st.session_state.generating,
        ):
            st.session_state.generating = True
            st.session_state.action = "merge_final"
            st.rerun()

        _task_handler(
            "merge_final",
            lambda: final_generation(
                st.session_state.video_data,
                use_custom_audio=st.session_state.use_custom_audio,
                final_quality=RESOLUTIONS[st.session_state.download_quality],
            ),
        )

    if st.session_state.get("video_generated") and st.session_state.get(
        "final_output_path"
    ):
        with st.expander("📽️ FINAL VIDEO"):
            st.video(st.session_state.final_output_path)
            with open(st.session_state.final_output_path, "rb") as f:
                st.download_button(
                    "DOWNLOAD VIDEO",
                    data=f.read(),
                    file_name="final_video.mp4",
                    mime="video/mp4",
                    width="stretch",
                )

    if st.session_state.rerun_needed:
        st.session_state.rerun_needed = False
