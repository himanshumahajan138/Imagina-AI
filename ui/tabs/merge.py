"""Merge Videos tab."""

from __future__ import annotations

import os
import tempfile
import uuid

import ffmpeg
import streamlit as st


def render() -> None:
    st.title("🎬 :red[**Merge Videos**] ➕")

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

    if uploaded_videos:
        st.session_state.merge_uploaded_videos = uploaded_videos
        file_names = [file.name for file in uploaded_videos]
        if not st.session_state.merge_video_order or set(
            st.session_state.merge_video_order
        ) != set(file_names):
            st.session_state.merge_video_order = file_names

    if st.session_state.merge_uploaded_videos:
        file_names = [file.name for file in st.session_state.merge_uploaded_videos]

        order = st.multiselect(
            "Select the order of videos (top = first):",
            options=file_names,
            default=st.session_state.merge_video_order,
        )
        if order:
            st.session_state.merge_video_order = order

        st.subheader("Transition Settings")
        col1, col2 = st.columns(2)
        with col1:
            transition_type = st.selectbox(
                "Transition Effect",
                options=[
                    "None", "fade", "fadeblack", "fadewhite",
                    "wipeleft", "wiperight", "wipeup", "wipedown",
                    "slideleft", "slideright", "slideup", "slidedown",
                    "circlecrop", "circleopen", "dissolve",
                ],
                help="Select the transition effect between videos",
            )
        with col2:
            transition_duration = st.slider(
                "Transition Duration (seconds)",
                min_value=0.0, max_value=3.0, value=0.5, step=0.1,
                disabled=(transition_type == "None"),
                help="Duration of the transition effect",
            )

        if st.button("Merge Videos", key="merge_videos_button", width="stretch"):
            with st.status("🔗 Merging videos... please wait"):
                try:
                    temp_dir = tempfile.gettempdir()
                    temp_paths = []
                    for name in st.session_state.merge_video_order:
                        file = next(
                            f for f in st.session_state.merge_uploaded_videos if f.name == name
                        )
                        file.seek(0)
                        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.name}")
                        with open(temp_path, "wb") as f:
                            f.write(file.read())
                        temp_paths.append(temp_path)

                    output_path = os.path.join(temp_dir, f"merged_{uuid.uuid4()}.mp4")

                    if transition_type == "None" or len(temp_paths) == 1:
                        input_streams = []
                        for path in temp_paths:
                            probe = ffmpeg.probe(path)
                            has_audio = any(s["codec_type"] == "audio" for s in probe["streams"])
                            inp = ffmpeg.input(path)
                            input_streams.append(inp.video)
                            if has_audio:
                                input_streams.append(inp.audio)

                        concat = ffmpeg.concat(*input_streams, v=1, a=1, unsafe=1).node
                        v = concat[0]
                        a = concat[1]

                        (
                            ffmpeg.output(
                                v, a, output_path,
                                vcodec="libx264", acodec="aac",
                                pix_fmt="yuv420p", movflags="+faststart",
                            )
                            .overwrite_output()
                            .run(capture_stdout=True, capture_stderr=True)
                        )
                    else:
                        inputs = []
                        for i, path in enumerate(temp_paths):
                            probe = ffmpeg.probe(path)
                            duration = float(probe["format"]["duration"])
                            inp = ffmpeg.input(path)
                            if i < len(temp_paths) - 1:
                                video_extended = ffmpeg.filter(
                                    inp.video, "tpad",
                                    stop_mode="clone", stop_duration=transition_duration,
                                )
                                audio_extended = ffmpeg.filter(
                                    inp.audio, "apad", pad_dur=transition_duration
                                )
                            else:
                                video_extended = inp.video
                                audio_extended = inp.audio
                            inputs.append(
                                {"video": video_extended, "audio": audio_extended, "duration": duration}
                            )

                        video = inputs[0]["video"]
                        audio = inputs[0]["audio"]
                        offset = inputs[0]["duration"]

                        for i in range(1, len(inputs)):
                            video = ffmpeg.filter(
                                [video, inputs[i]["video"]],
                                "xfade",
                                transition=transition_type,
                                duration=transition_duration,
                                offset=offset,
                            )
                            audio = ffmpeg.filter(
                                [audio, inputs[i]["audio"]],
                                "acrossfade",
                                d=transition_duration,
                            )
                            offset += inputs[i]["duration"]

                        (
                            ffmpeg.output(
                                video, audio, output_path,
                                vcodec="libx264", acodec="aac",
                                pix_fmt="yuv420p", movflags="+faststart",
                            )
                            .overwrite_output()
                            .run(capture_stdout=True, capture_stderr=True)
                        )

                    st.session_state.merged_output_path = output_path
                    st.success("✅ Merged video created successfully!")

                except ffmpeg.Error as e:
                    st.error("⚠️ FFmpeg failed while merging videos.")
                    st.text(e.stderr.decode() if e.stderr else str(e))
                    st.session_state.merged_output_path = None
                except Exception as e:
                    st.error(f"⚠️ An error occurred: {str(e)}")
                    st.session_state.merged_output_path = None

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
