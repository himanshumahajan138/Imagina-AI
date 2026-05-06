"""Heavy Streamlit gallery components for the cinematic flow.

Hosts `storyboard_gallery` (per-scene image preview/regeneration) and
`video_gallery` (per-scene generated-video preview/regeneration) so the
cinematic tab body stays small.
"""

from __future__ import annotations

import base64
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import streamlit as st
from PIL import Image

from pipelines.cinematic import _generate_audio, _generate_single_video, _generate_storyboard_images


def storyboard_gallery(
    global_dimension: str,
    script: List[Dict],
    model_type: str,
    use_custom_audio: bool,
):
    with st.expander("🖼️ Storyboard Gallery", expanded=True):
        if use_custom_audio:
            with st.expander("🎧 Generated Audio"):
                if "new_audio_data" in st.session_state:
                    st.write("Original Audio")
                    audio_path = st.session_state.video_data.get("audio_path", None)
                    if audio_path:
                        st.audio(audio_path)
                    st.divider()
                    st.write("New Generated Audio")
                    new_audio_data = st.session_state.new_audio_data
                    st.audio(new_audio_data["path"])
                    st.divider()
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button(
                            "REPLACE AUDIO",
                            key="replace_audio",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.video_data["audio_path"] = new_audio_data["path"]
                            st.session_state.pop("new_audio_data", None)
                            st.rerun()

                    with col2:
                        if st.button(
                            "CANCEL",
                            key="cancel_audio",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.pop("new_audio_data", None)
                            st.rerun()
                else:
                    audio_path = st.session_state.video_data.get("audio_path", None)
                    if audio_path:
                        st.audio(audio_path)

                    if st.button(
                        "REGENERATE AUDIO",
                        key="regen_audio",
                        width="stretch",
                        disabled=st.session_state.generating,
                    ):
                        with st.spinner("Regenerating audio..."):
                            try:
                                new_audio_data = _generate_audio(
                                    script, st.session_state.get("custom_bgm")
                                )
                                st.session_state.new_audio_data = new_audio_data
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to Regenerate Audio: {e}")

        if "new_image_data" not in st.session_state:
            st.session_state.new_image_data = {}

        for i, entry in enumerate(st.session_state.video_data["images"]):
            with st.expander(f"🎬 Scene {i + 1}"):
                st.text_area("📜 Script", value=entry["script"], disabled=True)
                st.text_area("🎥 Scene Description", value=entry["scene"], disabled=True)
                st.text_area(
                    "🎥 Video Scene Description",
                    value=entry["video_scene"],
                    disabled=True,
                )

                if entry.get("custom_image"):
                    st.image(entry["custom_image"], caption="Custom Image", width="stretch")
                else:
                    st.image(entry["image_path"], caption="Generated Image", width="stretch")

                with st.expander("REPLACE IMAGE"):
                    uploaded = st.file_uploader(
                        f"Replace Scene {i + 1} Image",
                        key=f"upload_{i}",
                        type=["png", "jpg", "jpeg"],
                        disabled=st.session_state.generating,
                    )

                    if uploaded:
                        with st.expander("Preview Uploaded Image"):
                            st.image(uploaded, caption="Uploaded Image", width="stretch")

                        if st.button(
                            f"Update Scene {i + 1} with This Image",
                            key=f"confirm_upload_{i}",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            temp_image_path = (
                                Path(tempfile.gettempdir()) / f"updated_image_{uuid.uuid4()}.png"
                            )
                            img_data = uploaded.read()
                            encoded_img = base64.b64encode(img_data).decode("utf-8")
                            image_data_url = f"data:image/png;base64,{encoded_img}"
                            image = Image.open(BytesIO(img_data))
                            image.save(temp_image_path)

                            st.session_state.video_data["images"][i]["custom_image"] = (
                                str(temp_image_path)
                            )
                            if i < len(st.session_state.script_df):
                                st.session_state.script_df.at[i, "custom_image"] = image_data_url

                            st.session_state.rerun_needed = True
                            st.rerun()

                if i in st.session_state.new_image_data:
                    st.divider()
                    st.write("New Generated Image:")
                    st.image(
                        st.session_state.new_image_data[i]["image_path"],
                        caption="New Generated Image",
                        width="stretch",
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            "REPLACE IMAGE",
                            key=f"replace_image_{i}",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.video_data["images"][i]["image_path"] = (
                                st.session_state.new_image_data[i]["image_path"]
                            )
                            st.session_state.video_data["images"][i]["custom_image"] = None
                            st.session_state.script_df.at[i, "custom_image"] = None
                            st.session_state.new_image_data.pop(i)
                            st.rerun()

                    with col2:
                        if st.button(
                            "CANCEL",
                            key=f"cancel_image_{i}",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.new_image_data.pop(i)
                            st.rerun()
                else:
                    custom_reference_imgs: list = []
                    custom_regenerate_text = ""
                    with st.expander("REGENRATE IMAGE WITH CUSTOM INSTRUCTIONS"):
                        if st.session_state.image_refinement_mode:
                            custom_reference_imgs = st.file_uploader(
                                "Upload Custom Reference Images (2 MAX)",
                                type="png",
                                accept_multiple_files=True,
                                key=f"custom_reference_imgs_{i}",
                            )
                            if custom_reference_imgs and len(custom_reference_imgs) > 2:
                                st.warning(
                                    "⚠️ You can upload up to 2 images only. "
                                    f"You uploaded {len(custom_reference_imgs)}."
                                )
                                custom_reference_imgs = custom_reference_imgs[:2]

                        custom_regenerate_text = st.text_area(
                            "Custom Image Generation Instructions",
                            placeholder="Enter image generation instructions here",
                            disabled=st.session_state.generating,
                            key=f"regenerate_text_{i}",
                            max_chars=300,
                        )

                        if st.button(
                            "REGENERATE IMAGE",
                            key=f"regenerate_{i}",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            with st.spinner(f"Regenerating Scene {i + 1}..."):
                                try:
                                    item = {
                                        "scene": entry["scene"],
                                        "video_scene": entry["video_scene"],
                                        "script": entry["script"],
                                        "custom_image": None,
                                        "reference_images": custom_reference_imgs,
                                        "reference_text": custom_regenerate_text,
                                        "old_image": (
                                            entry.get("custom_image")
                                            if entry.get("custom_image")
                                            else entry.get("image_path")
                                        ),
                                    }
                                    new_img_gen = _generate_storyboard_images(
                                        [item], global_dimension, model_type
                                    )
                                    new_img = next(new_img_gen)
                                    st.session_state.new_image_data[i] = {
                                        "image_path": new_img["image_path"]
                                    }
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed to regenerate image: {e}")


def video_gallery(global_dimension, model_type):
    if "new_video_data" not in st.session_state:
        st.session_state.new_video_data = {}
    with st.expander("Preview Generated Scene Videos", expanded=True):
        for scene, value in st.session_state.scene_videos.items():
            with st.expander(f"Scene {scene}"):
                st.text_area(
                    "Script",
                    value["script"],
                    disabled=True,
                    key=f"generated_video_script_{scene}",
                )
                st.text_area(
                    "Scene Text",
                    value["scene_text"],
                    disabled=True,
                    key=f"generated_video_{scene}",
                )
                st.text_area(
                    "Video Scene Text",
                    value["video_scene_text"],
                    disabled=True,
                    key=f"generated_video_scene_{scene}",
                )
                video_path = value["video_path"]
                if video_path:
                    st.video(video_path)
                else:
                    st.error("❌ Video generation failed for this scene.")

                if scene in st.session_state.new_video_data:
                    st.divider()
                    st.write("New Regenerated Scene Video:")
                    new_video_path = st.session_state.new_video_data[scene]["video_path"]
                    if new_video_path:
                        st.video(new_video_path)
                    else:
                        st.error("❌ Video generation failed for this scene.")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            "REPLACE VIDEO",
                            key=f"replace_video_{scene}",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.scene_videos[scene]["video_path"] = (
                                st.session_state.new_video_data[scene]["video_path"]
                            )
                            st.session_state.new_video_data.pop(scene)
                            st.rerun()

                    with col2:
                        if st.button(
                            "CANCEL",
                            key=f"cancel_video_{scene}",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.new_video_data.pop(scene)
                            st.rerun()
                else:
                    if st.button(
                        "REGENERATE SCENE VIDEO",
                        key=f"regen_video_{scene}",
                        width="stretch",
                        disabled=st.session_state.generating,
                    ):
                        with st.spinner("Regenerating scene video..."):
                            try:
                                scene_data = st.session_state.scene_video_data[scene - 1]
                                new_file = _generate_single_video(
                                    scene_data, "", model_type, global_dimension
                                )
                                st.session_state.new_video_data[scene] = {
                                    "video_path": new_file,
                                    "script": scene_data["script"],
                                    "scene_text": scene_data["scene_text"],
                                    "video_scene_text": scene_data["video_scene_text"],
                                }
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to regenerate scene {scene} video: {e}")
