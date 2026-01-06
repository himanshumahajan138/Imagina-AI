# streamlit run app.py --server.address 0.0.0.0 --server.port 8004 --server.enableCORS false
import os
import cv2
import uuid
import ffmpeg
import base64
import tempfile
import pandas as pd
import streamlit as st
from streamlit_theme import st_theme
from core.utils import (
    hash_df,
    generate_video,
    generate_audio_images,
    openai_script_generator,
    validate_script_data,
    parse_script_scene_content,
    storyboard_gallery,
    final_generation,
    video_gallery,
    extract_frame,
    remove_watermark_ffmpeg,
    remove_watermark_opencv,
)
from core.trimmer_utils import (
    save_uploaded_file,
    get_file_duration,
    display_file_info,
    display_media_player,
    display_timeline,
    trim_media,
    format_time,
)
from core.yt_downloader_utils import is_valid_youtube_url, youtube_downloader_pipeline
from core.config import (
    SPEAKER_OPTIONS,
    DIMENSIONS,
    MODEL_TYPES,
    COMMON_LANGUAGES,
    RESOLUTIONS,
)

# Flags
for key in ["rerun_needed", "scene_videos_generated", "generating", "action"]:
    if key not in st.session_state:
        st.session_state[key] = False

st.set_page_config(page_title="🎬 Imagina AI Video Generator", layout="wide")

streamlit_theme = st_theme(key="streamlit_theme")
# Check if theme exists and if it's light (white background)
if streamlit_theme and streamlit_theme.get("base") == "light":
    logo_path = "images\\logo-white.png"  # white logo for light theme
else:
    logo_path = "images\\logo-black.png"  # black logo for dark theme (default to dark)

# col1, col2, col3 = st.columns([1, 1, 1])
# with col2:
#     st.image(logo_path, width=250)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎬 Cinematic Generator",
        "➕ Merge Videos",
        "✂️ Video Watermark Remover",
        "✂️ Media Trimmer",
        "📽️ YouTube Downloader",
    ]
)


# ========== Sidebar Configuration ==========
with st.sidebar:
    st.sidebar.image(logo_path)
    st.header("🎛️ Global Settings", divider="red")

    with st.expander("🎤 Voice & Video Settings", expanded=True):
        model_label = st.selectbox(
            "Model Type",
            MODEL_TYPES.keys(),
            index=0,
            disabled=st.session_state.generating,
        )
        st.session_state.model_type = MODEL_TYPES[model_label]
        st.session_state.language = st.selectbox(
            "Language",
            COMMON_LANGUAGES,
            index=(
                COMMON_LANGUAGES.index("english")
                if "english" in COMMON_LANGUAGES
                else 0
            ),
            disabled=st.session_state.generating,
        )
        st.session_state.dimension = st.selectbox(
            "Video Dimensions", DIMENSIONS.keys(), disabled=st.session_state.generating
        )
        st.session_state.download_quality = st.selectbox(
            "Download Resolution",
            RESOLUTIONS.keys(),
            index=1,
            disabled=st.session_state.generating,
        )

        st.session_state.duration = st.slider(
            "Duration (seconds)",
            (
                8
                if st.session_state.model_type == "gemini"
                else 12 if st.session_state.model_type == "openai" else 10
            ),
            (
                120
                if st.session_state.model_type == "gemini"
                else 180 if st.session_state.model_type == "openai" else 150
            ),
            (
                16
                if st.session_state.model_type == "gemini"
                else 24 if st.session_state.model_type == "openai" else 20
            ),
            step=(
                8
                if st.session_state.model_type == "gemini"
                else 12 if st.session_state.model_type == "openai" else 10
            ),
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
                if custom_reference_imgs:
                    if len(custom_reference_imgs) > 3:
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
                    index=list(SPEAKER_OPTIONS.keys()).index("Wayne"),
                    disabled=st.session_state.generating,
                    key="selected_speaker",
                )
                st.slider(
                    "Speed",
                    0.0,
                    2.0,
                    1.0,
                    step=0.1,
                    disabled=st.session_state.generating,
                    key="selected_speed",
                )
                st.slider(
                    "Pitch",
                    -16,
                    16,
                    0,
                    step=1,
                    disabled=st.session_state.generating,
                    key="selected_pitch",
                )
                st.file_uploader(
                    "Upload Custom BGM (.wav)",
                    type="wav",
                    disabled=st.session_state.generating,
                    key="custom_bgm",
                )
        else:
            st.session_state.selected_speaker = list(SPEAKER_OPTIONS.keys()).index(
                "Wayne"
            )
            st.session_state.selected_speed = None
            st.session_state.selected_pitch = None

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
            disabled=st.session_state.generating
            or st.session_state.script_file is None,
        )

# ==================== TAB 1: CINEMATIC GENERATOR ====================
with tab1:
    st.title(":clapper: :red[**Cinematic Theme to Video Generator**] :clapper:")

    # Function to handle task execution pattern
    def task_handler(action_name, task_func):
        if (
            st.session_state.get("generating")
            and st.session_state.get("action") == action_name
        ):
            task_func()
            st.session_state.generating = False
            st.session_state.action = None
            st.rerun()

    # ========== Triggered Script Generations ==========
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

    task_handler(
        "generate_script",
        lambda: st.session_state.update(
            {
                "script_df": pd.DataFrame(
                    openai_script_generator(
                        st.session_state.theme,
                        st.session_state.language,
                        st.session_state.duration,
                        model_type=st.session_state.model_type,
                    )
                ).assign(
                    speed=st.session_state.selected_speed,
                    pitch=st.session_state.selected_pitch,
                    speaker=st.session_state.selected_speaker,
                    custom_image=None,
                )
            }
        ),
    )

    task_handler(
        "load_script",
        lambda: st.session_state.update(
            {
                "script_df": pd.DataFrame(
                    parse_script_scene_content(st.session_state.uploaded_content)
                ).assign(
                    speed=st.session_state.selected_speed,
                    pitch=st.session_state.selected_pitch,
                    speaker=st.session_state.selected_speaker,
                    custom_image=None,
                ),
                "uploaded_content": None,
            }
        ),
    )

    # ========== Session State Script ==========
    if "script_df" not in st.session_state:
        st.session_state.script_df = pd.DataFrame(
            columns=[
                "script",
                "scene",
                "video_scene",
                "start_time",
                "end_time",
                "speed",
                "pitch",
                "speaker",
                "custom_image",
            ]
        )

    # ========== Editable Table ==========
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
                "pitch": st.column_config.NumberColumn(
                    "Pitch",
                    min_value=-16,
                    max_value=16,
                    step=1,
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
                                st.session_state.custom_image_upload,
                                width="stretch",
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

    # ========== Generate Scenes and Audio ==========
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

    task_handler("generate_audio_image", generate_audio_image_task)

    # ========== Show Storyboard Gallery ==========
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

    task_handler(
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

        task_handler(
            "merge_final",
            lambda: final_generation(
                st.session_state.video_data,
                use_custom_audio=st.session_state.use_custom_audio,
                final_quality=RESOLUTIONS[st.session_state.download_quality],
            ),
        )

    # ========== Show Final Video ==========
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

# ==================== TAB 2: MERGE VIDEOS ====================
with tab2:
    st.title("🎬 :red[**Merge Videos**] ➕")

    # Initialize session state variables
    if "merge_uploaded_videos" not in st.session_state:
        st.session_state.merge_uploaded_videos = []
    if "merge_video_order" not in st.session_state:
        st.session_state.merge_video_order = []
    if "merged_output_path" not in st.session_state:
        st.session_state.merged_output_path = None

    uploaded_videos = st.file_uploader(
        "Upload your video clips",
        type=["mp4", "mov", "avi"],
        accept_multiple_files=True,
        key="merge_videos_uploader",
    )

    # Store uploaded videos in session state
    if uploaded_videos:
        st.session_state.merge_uploaded_videos = uploaded_videos
        file_names = [file.name for file in uploaded_videos]

        # Initialize order if not set or if files changed
        if not st.session_state.merge_video_order or set(
            st.session_state.merge_video_order
        ) != set(file_names):
            st.session_state.merge_video_order = file_names

    # Only show controls if videos are uploaded
    if st.session_state.merge_uploaded_videos:
        file_names = [file.name for file in st.session_state.merge_uploaded_videos]

        # Let user choose order manually
        order = st.multiselect(
            "Select the order of videos (top = first):",
            options=file_names,
            default=st.session_state.merge_video_order,
        )

        # Update order in session state
        if order:
            st.session_state.merge_video_order = order

        # Transition options
        st.subheader("Transition Settings")
        col1, col2 = st.columns(2)

        with col1:
            transition_type = st.selectbox(
                "Transition Effect",
                options=[
                    "None",
                    "fade",
                    "fadeblack",
                    "fadewhite",
                    "wipeleft",
                    "wiperight",
                    "wipeup",
                    "wipedown",
                    "slideleft",
                    "slideright",
                    "slideup",
                    "slidedown",
                    "circlecrop",
                    "circleopen",
                    "dissolve",
                ],
                help="Select the transition effect between videos",
            )

        with col2:
            transition_duration = st.slider(
                "Transition Duration (seconds)",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.1,
                disabled=(transition_type == "None"),
                help="Duration of the transition effect",
            )

        if st.button("Merge Videos", key="merge_videos_button", width="stretch"):
            with st.status("🔗 Merging videos... please wait"):
                try:
                    temp_dir = tempfile.gettempdir()
                    temp_paths = []

                    # Save uploaded files locally in temp dir
                    for name in st.session_state.merge_video_order:
                        file = next(
                            f
                            for f in st.session_state.merge_uploaded_videos
                            if f.name == name
                        )
                        # Reset file pointer to beginning
                        file.seek(0)
                        temp_path = os.path.join(
                            temp_dir, f"{uuid.uuid4()}_{file.name}"
                        )
                        with open(temp_path, "wb") as f:
                            f.write(file.read())
                        temp_paths.append(temp_path)

                    output_path = os.path.join(temp_dir, f"merged_{uuid.uuid4()}.mp4")

                    if transition_type == "None" or len(temp_paths) == 1:
                        # No transitions - use simple concatenation
                        input_streams = []

                        for path in temp_paths:
                            probe = ffmpeg.probe(path)
                            has_audio = any(
                                s["codec_type"] == "audio" for s in probe["streams"]
                            )
                            inp = ffmpeg.input(path)
                            input_streams.append(inp.video)
                            if has_audio:
                                input_streams.append(inp.audio)

                        concat = ffmpeg.concat(*input_streams, v=1, a=1, unsafe=1).node
                        v = concat[0]
                        a = concat[1]

                        (
                            ffmpeg.output(
                                v,
                                a,
                                output_path,
                                vcodec="libx264",
                                acodec="aac",
                                pix_fmt="yuv420p",
                                movflags="+faststart",
                            )
                            .overwrite_output()
                            .run(capture_stdout=True, capture_stderr=True)
                        )
                    else:
                        # With transitions - extend videos to preserve total duration
                        inputs = []

                        for i, path in enumerate(temp_paths):
                            probe = ffmpeg.probe(path)
                            duration = float(probe["format"]["duration"])

                            inp = ffmpeg.input(path)

                            # Extend each video (except last) by transition_duration
                            if i < len(temp_paths) - 1:
                                # Freeze last frame for transition_duration
                                video_extended = ffmpeg.filter(
                                    inp.video,
                                    "tpad",
                                    stop_mode="clone",
                                    stop_duration=transition_duration,
                                )

                                # Extend audio with silence
                                audio_extended = ffmpeg.filter(
                                    inp.audio, "apad", pad_dur=transition_duration
                                )
                            else:
                                video_extended = inp.video
                                audio_extended = inp.audio

                            inputs.append(
                                {
                                    "video": video_extended,
                                    "audio": audio_extended,
                                    "duration": duration,
                                }
                            )

                        # Start with first video
                        video = inputs[0]["video"]
                        audio = inputs[0]["audio"]

                        # Calculate offset for transitions
                        offset = inputs[0]["duration"]

                        # Apply transitions between consecutive videos
                        for i in range(1, len(inputs)):
                            # Apply xfade transition
                            video = ffmpeg.filter(
                                [video, inputs[i]["video"]],
                                "xfade",
                                transition=transition_type,
                                duration=transition_duration,
                                offset=offset,
                            )

                            # Mix audio with cross-fade at the right position
                            audio = ffmpeg.filter(
                                [audio, inputs[i]["audio"]],
                                "acrossfade",
                                d=transition_duration,
                            )

                            # Update offset for next transition
                            offset += inputs[i]["duration"]

                        (
                            ffmpeg.output(
                                video,
                                audio,
                                output_path,
                                vcodec="libx264",
                                acodec="aac",
                                pix_fmt="yuv420p",
                                movflags="+faststart",
                            )
                            .overwrite_output()
                            .run(capture_stdout=True, capture_stderr=True)
                        )

                    # Store output path in session state
                    st.session_state.merged_output_path = output_path

                    st.success("✅ Merged video created successfully!")

                except ffmpeg.Error as e:
                    st.error("⚠️ FFmpeg failed while merging videos.")
                    st.text(e.stderr.decode() if e.stderr else str(e))
                    st.session_state.merged_output_path = None
                except Exception as e:
                    st.error(f"⚠️ An error occurred: {str(e)}")
                    st.session_state.merged_output_path = None

        # Display merged video if it exists in session state
        if st.session_state.merged_output_path and os.path.exists(
            st.session_state.merged_output_path
        ):
            st.video(st.session_state.merged_output_path)

            with open(st.session_state.merged_output_path, "rb") as video_file:
                st.download_button(
                    "Download Merged Video",
                    data=video_file,
                    file_name="merged_video.mp4",
                    mime="video/mp4",
                    key="download_merged_video",
                    width="stretch",
                    type="primary",
                )

# ==================== TAB 3: WATERMARK REMOVER ====================
with tab3:
    st.title("🎬 :red[**Video Watermark Remover**] ✂️")

    # Initialize session state
    if "processed_video_path" not in st.session_state:
        st.session_state.processed_video_path = None
    if "original_video_path" not in st.session_state:
        st.session_state.original_video_path = None
    if "coordinates" not in st.session_state:
        st.session_state.coordinates = None
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False

    uploaded_file = st.file_uploader(
        "Upload your video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file:
        # Save uploaded file temporarily (only once)
        if st.session_state.original_video_path is None or not os.path.exists(
            st.session_state.original_video_path
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.original_video_path = tmp_file.name

        temp_input_path = st.session_state.original_video_path

        # Extract and display first frame
        st.subheader("📍 Step 1: Select Watermark Area")
        frame = extract_frame(temp_input_path)

        if frame is not None:
            st.markdown("### Watermark Position")

            # Get video dimensions
            height, width = frame.shape[:2]
            st.write(f"**Video Size:** {width} x {height} px")

            st.markdown("---")

            # Preset corner options
            st.markdown("**⚙️ Settings**")

            # Method selection
            removal_method = st.radio(
                "Removal Method:",
                [
                    "OpenCV Inpainting (Better Quality, Slower)",
                    "FFmpeg Delogo (Faster, More Blur)",
                ],
                help="OpenCV provides more natural results but takes longer",
                key="method_select",
            )

            # Inpainting algorithm (only for OpenCV)
            if "OpenCV" in removal_method:
                inpaint_algo = st.selectbox(
                    "Inpainting Algorithm:",
                    ["Telea (Better for textures)", "Navier-Stokes (Better for edges)"],
                    help="Telea works best for textured backgrounds, Navier-Stokes for smooth gradients",
                    key="inpaint_algo",
                )

            corner = st.selectbox(
                "Choose corner:",
                ["Custom", "Bottom Right", "Bottom Left", "Top Right", "Top Left"],
                key="corner_select",
            )

            # Default watermark size
            wm_width = st.number_input(
                "Watermark Width (px)", 50, width, 50, key="wm_width"
            )
            wm_height = st.number_input(
                "Watermark Height (px)", 20, height, 25, key="wm_height"
            )
            margin = st.number_input("Margin from edge (px)", 0, 100, 30, key="margin")

            # Calculate coordinates based on corner selection
            if corner == "Bottom Right":
                x = width - wm_width - margin
                y = height - wm_height - margin
            elif corner == "Bottom Left":
                x = margin
                y = height - wm_height - margin
            elif corner == "Top Right":
                x = width - wm_width - margin
                y = margin
            elif corner == "Top Left":
                x = margin
                y = margin
            else:  # Custom
                st.markdown("**Manual Entry:**")
                x = st.number_input("X (left position)", 0, width, 0, key="x_pos")
                y = st.number_input("Y (top position)", 0, height, 0, key="y_pos")

            # Store coordinates in session state
            st.session_state.coordinates = (x, y, wm_width, wm_height)

            st.markdown("---")
            st.markdown(
                f"**Selected Region: | X: {int(x)} | Y: {int(y)} | Width: {wm_width} | Height: {wm_height} |**"
            )

            # Show preview with rectangle
            preview_frame = frame.copy()
            cv2.rectangle(
                preview_frame,
                (int(x), int(y)),
                (int(x + wm_width), int(y + wm_height)),
                (255, 0, 0),
                3,
            )
            st.image(preview_frame, caption="Preview with Selection", width="stretch")

        # Process button
        st.markdown("---")
        if st.button("Remove Watermark", type="primary", width="stretch"):
            with st.spinner("Processing video... This may take a few minutes."):
                try:
                    # Create output file
                    output_path = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".mp4"
                    ).name

                    # Choose removal method
                    if "OpenCV" in removal_method:
                        method = "telea" if "Telea" in inpaint_algo else "ns"
                        remove_watermark_opencv(
                            temp_input_path,
                            output_path,
                            int(x),
                            int(y),
                            int(wm_width),
                            int(wm_height),
                            method,
                        )
                    else:
                        remove_watermark_ffmpeg(
                            temp_input_path,
                            output_path,
                            int(x),
                            int(y),
                            int(wm_width),
                            int(wm_height),
                        )

                    # Store processed video path in session state
                    st.session_state.processed_video_path = output_path
                    st.session_state.processing_done = True

                    st.success("✅ Watermark removed successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    st.info(
                        "Make sure FFmpeg and OpenCV are installed: `pip install ffmpeg-python opencv-python numpy`"
                    )

        # Show processed video preview and download
        if st.session_state.processing_done and st.session_state.processed_video_path:
            st.markdown("---")
            st.subheader("✨ Step 2: Preview & Download")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🎥 Original Video**")
                if os.path.exists(st.session_state.original_video_path):
                    st.video(st.session_state.original_video_path)

            with col2:
                st.markdown("**✅ Processed Video (Watermark Removed)**")
                if os.path.exists(st.session_state.processed_video_path):
                    st.video(st.session_state.processed_video_path)

            st.markdown("---")

            # Download button
            if os.path.exists(st.session_state.processed_video_path):
                with open(st.session_state.processed_video_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f,
                        file_name="video_no_watermark.mp4",
                        mime="video/mp4",
                        width="stretch",
                    )

            # Reset button
            if st.button("🔄 Process Another Video", width="stretch"):
                # Cleanup temporary files
                if st.session_state.processed_video_path and os.path.exists(
                    st.session_state.processed_video_path
                ):
                    os.unlink(st.session_state.processed_video_path)
                if st.session_state.original_video_path and os.path.exists(
                    st.session_state.original_video_path
                ):
                    os.unlink(st.session_state.original_video_path)

                # Reset session state
                st.session_state.processed_video_path = None
                st.session_state.original_video_path = None
                st.session_state.coordinates = None
                st.session_state.processing_done = False
                st.rerun()

    else:
        st.info("👆 Upload a video to get started")

        st.markdown(
            """
        ### How to use:
        1. **Upload your video** with the watermark
        2. **Choose removal method** - OpenCV for better quality or FFmpeg for speed
        3. **Select watermark position** - Choose a corner preset or enter custom coordinates
        4. **Adjust size** - Set the width and height of the watermark area
        5. **Preview** - Check the red rectangle covers your watermark
        6. **Remove** - Click the button and download your cleaned video!
        
        ### Method Comparison:
        - **OpenCV Inpainting**: More natural results, reconstructs textures intelligently
          - *Telea*: Best for textured/patterned backgrounds
          - *Navier-Stokes*: Best for smooth gradients and edges
        - **FFmpeg Delogo**: Faster processing but creates more blur
        
        ### Tips:
        - Use the **corner presets** for watermarks in standard positions
        - Try **OpenCV Inpainting** first for best quality
        - Adjust the **margin** if watermark isn't exactly at the edge
        - Make the selection **slightly larger** than the watermark for best results
        - Test with a short video first to verify settings
        """
        )

# ==================== TAB 3: Media Trimmer ====================
with tab4:
    st.title("🎬 :red[**Media Trimmer**] ✂️")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("📁 Upload Media File")
        uploaded_file = st.file_uploader(
            "Choose an audio or video file",
            type=[
                "mp4",
                "mov",
                "avi",
                "mkv",
                "webm",
                "flv",
                "mp3",
                "wav",
                "aac",
                "flac",
                "m4a",
                "wma",
                "ogg",
            ],
        )

    with col2:
        st.subheader("ℹ️ Supported Formats")
        st.markdown(
            """
        **Video:** MP4, MOV, AVI, MKV, WEBM, FLV
        
        **Audio:** MP3, WAV, AAC, FLAC, M4A, WMA, OGG
        """
        )

    if uploaded_file:
        # Save and get duration
        with st.spinner("Processing file..."):
            file_path = save_uploaded_file(uploaded_file)
            duration = get_file_duration(file_path)

        if duration > 0:
            st.success(
                f"✅ File loaded successfully | Duration: {format_time(duration)}"
            )

            # Display file info
            st.subheader("📋 File Information")
            display_file_info(file_path, uploaded_file.name)
            # Display media player
            display_media_player(file_path, uploaded_file.name)
            # Trim controls
            st.subheader("✂️ Set Trim Range")

            col1, col2 = st.columns(2)

            with col1:
                start_time = st.slider(
                    "Start Time (seconds)",
                    min_value=0.0,
                    max_value=duration,
                    value=0.0,
                    step=0.1,
                    key="start_slider",
                )

            with col2:
                end_time = st.slider(
                    "End Time (seconds)",
                    min_value=0.0,
                    max_value=duration,
                    value=duration,
                    step=0.1,
                    key="end_slider",
                )

            # Update timeline visualization
            if start_time < end_time:
                display_timeline(duration, start_time, end_time)
            else:
                st.warning("⚠️ Start time must be less than end time")

            # Preview
            st.subheader("👀 Preview Range")
            preview_col1, preview_col2, preview_col3 = st.columns(3)
            with preview_col1:
                st.write(f"**Start:** {format_time(start_time)}")
            with preview_col2:
                st.write(f"**Trim Length:** {format_time(end_time - start_time)}")
            with preview_col3:
                st.write(f"**End:** {format_time(end_time)}")

            if st.button("✂️ Trim", width="stretch"):
                if start_time >= end_time:
                    st.error("Invalid time range! Start time must be before end time.")
                else:
                    progress_placeholder = st.empty()
                    output_path = os.path.join(
                        tempfile.gettempdir(), f"trimmed_{uploaded_file.name}"
                    )

                    if trim_media(
                        file_path,
                        start_time,
                        end_time,
                        output_path,
                        progress_placeholder,
                    ):

                        # Get output file info
                        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)

                        progress_placeholder.success(
                            f"✅ File trimmed successfully | Output file size: {output_size_mb:.2f} MB"
                        )
                        display_media_player(
                            output_path, f"trimmed_{uploaded_file.name}"
                        )
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Trimmed File",
                                data=f.read(),
                                file_name=f"trimmed_{uploaded_file.name}",
                                width="stretch",
                                type="primary",
                            )

                        # Cleanup
                        if os.path.exists(output_path):
                            os.remove(output_path)

            # Cleanup temp file
            if os.path.exists(file_path):
                os.remove(file_path)

# ==================== TAB 3: Youtube Downloader ====================
with tab5:
    st.title("🎬 :red[**YouTube Downloader**] 📽️")

    # Initialize session state
    if "download_result" not in st.session_state:
        st.session_state.download_result = None
    if "result_type" not in st.session_state:
        st.session_state.result_type = None

    youtube_url = st.text_input(
        "📌 Enter YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste the full YouTube URL here",
    )

    # Download options
    st.subheader("⬇️ Download Options")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🎵 Audio Only", use_container_width=True, key="btn_audio"):
            if not youtube_url:
                st.error("❌ Please enter a YouTube URL")
            elif not is_valid_youtube_url(youtube_url):
                st.error("❌ Please enter a valid YouTube URL")
            else:
                with st.spinner("Downloading audio..."):
                    result = youtube_downloader_pipeline(youtube_url, "audio")
                    st.session_state.download_result = result
                    st.session_state.result_type = "audio"

    with col2:
        if st.button("🎬 Video Only", use_container_width=True, key="btn_video"):
            if not youtube_url:
                st.error("❌ Please enter a YouTube URL")
            elif not is_valid_youtube_url(youtube_url):
                st.error("❌ Please enter a valid YouTube URL")
            else:
                with st.spinner("Downloading video..."):
                    result = youtube_downloader_pipeline(youtube_url, "video")
                    st.session_state.download_result = result
                    st.session_state.result_type = "video"

    with col3:
        if st.button("🎬🎵 Mix", use_container_width=True, key="btn_mix"):
            if not youtube_url:
                st.error("❌ Please enter a YouTube URL")
            elif not is_valid_youtube_url(youtube_url):
                st.error("❌ Please enter a valid YouTube URL")
            else:
                with st.spinner("Downloading mix..."):
                    result = youtube_downloader_pipeline(youtube_url, "mix")
                    st.session_state.download_result = result
                    st.session_state.result_type = "mix"

    with col4:
        if st.button("📦 Both", use_container_width=True, key="btn_both"):
            if not youtube_url:
                st.error("❌ Please enter a YouTube URL")
            elif not is_valid_youtube_url(youtube_url):
                st.error("❌ Please enter a valid YouTube URL")
            else:
                with st.spinner("Downloading both files..."):
                    result = youtube_downloader_pipeline(youtube_url, "both")
                    st.session_state.download_result = result
                    st.session_state.result_type = "both"

    # Display results outside columns
    st.divider()

    if st.session_state.download_result:
        result = st.session_state.download_result
        result_type = st.session_state.result_type

        if not result["success"]:
            st.error(result["error"])
        else:
            if result_type == "audio":
                st.success("✅ Audio extracted successfully!")
                audio_path = result["data"].get("audio_path")
                if audio_path:
                    try:
                        display_media_player(audio_path, audio_path)
                        with open(audio_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download Audio",
                                data=f,
                                file_name=audio_path.split("/")[-1],
                                mime="audio/mpeg",
                                use_container_width=True,
                            )
                    except Exception as e:
                        st.warning(f"Could not load preview: {str(e)}")

            elif result_type == "video":
                st.success("✅ Video extracted successfully!")
                video_path = result["data"].get("video_path")
                if video_path:
                    try:
                        display_media_player(video_path, video_path)
                        with open(video_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download Video",
                                data=f,
                                file_name=video_path.split("/")[-1],
                                mime="video/mp4",
                                use_container_width=True,
                            )
                    except Exception as e:
                        st.warning(f"Could not load preview: {str(e)}")

            elif result_type == "mix":
                st.success("✅ Mix extracted successfully!")
                hybrid_path = result["data"].get("hybrid_path")
                if hybrid_path:
                    try:
                        display_media_player(hybrid_path, hybrid_path)
                        with open(hybrid_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download Mix",
                                data=f,
                                file_name=hybrid_path.split("/")[-1],
                                mime="video/mp4",
                                use_container_width=True,
                            )
                    except Exception as e:
                        st.warning(f"Could not load preview: {str(e)}")

            elif result_type == "both":
                st.success("✅ Both files extracted successfully!")
                audio_path = result["data"].get("audio_path")
                video_path = result["data"].get("video_path")

                st.subheader("🎵 Audio File")
                if audio_path:
                    try:
                        display_media_player(audio_path, audio_path)
                        with open(audio_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download Audio",
                                data=f,
                                file_name=audio_path.split("/")[-1],
                                mime="audio/mpeg",
                                use_container_width=True,
                                key="download_audio_both",
                            )
                    except Exception as e:
                        st.warning(f"Could not load audio preview: {str(e)}")

                st.subheader("🎬 Video File")
                if video_path:
                    try:
                        display_media_player(video_path, video_path)
                        with open(video_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download Video",
                                data=f,
                                file_name=video_path.split("/")[-1],
                                mime="video/mp4",
                                use_container_width=True,
                                key="download_video_both",
                            )
                    except Exception as e:
                        st.warning(f"Could not load video preview: {str(e)}")
