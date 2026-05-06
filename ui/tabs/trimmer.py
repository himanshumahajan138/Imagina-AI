"""Media Trimmer tab."""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from services.media.trimmer import (
    display_file_info,
    display_media_player,
    display_timeline,
    format_time,
    get_file_duration,
    save_uploaded_file,
    trim_media,
)


def render() -> None:
    st.title("🎬 :red[**Media Trimmer**] ✂️")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("📁 Upload Media File")
        uploaded_file = st.file_uploader(
            "Choose an audio or video file",
            type=[
                "mp4", "mov", "avi", "mkv", "webm", "flv",
                "mp3", "wav", "aac", "flac", "m4a", "wma", "ogg",
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

    if not uploaded_file:
        return

    with st.spinner("Processing file..."):
        file_path = save_uploaded_file(uploaded_file)
        duration = get_file_duration(file_path)

    if duration <= 0:
        return

    st.success(f"✅ File loaded successfully | Duration: {format_time(duration)}")

    st.subheader("📋 File Information")
    display_file_info(file_path, uploaded_file.name)
    display_media_player(file_path, uploaded_file.name)

    st.subheader("✂️ Set Trim Range")

    col1, col2 = st.columns(2)
    with col1:
        start_time = st.slider(
            "Start Time (seconds)",
            min_value=0.0, max_value=duration, value=0.0, step=0.1,
            key="start_slider",
        )
    with col2:
        end_time = st.slider(
            "End Time (seconds)",
            min_value=0.0, max_value=duration, value=duration, step=0.1,
            key="end_slider",
        )

    if start_time < end_time:
        display_timeline(duration, start_time, end_time)
    else:
        st.warning("⚠️ Start time must be less than end time")

    st.subheader("👀 Preview Range")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.write(f"**Start:** {format_time(start_time)}")
    with p2:
        st.write(f"**Trim Length:** {format_time(end_time - start_time)}")
    with p3:
        st.write(f"**End:** {format_time(end_time)}")

    if st.button("✂️ Trim", width="stretch"):
        if start_time >= end_time:
            st.error("Invalid time range! Start time must be before end time.")
        else:
            progress_placeholder = st.empty()
            output_path = os.path.join(tempfile.gettempdir(), f"trimmed_{uploaded_file.name}")

            if trim_media(file_path, start_time, end_time, output_path, progress_placeholder):
                output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                progress_placeholder.success(
                    f"✅ File trimmed successfully | Output file size: {output_size_mb:.2f} MB"
                )
                display_media_player(output_path, f"trimmed_{uploaded_file.name}")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Trimmed File",
                        data=f.read(),
                        file_name=f"trimmed_{uploaded_file.name}",
                        width="stretch",
                        type="primary",
                    )
                if os.path.exists(output_path):
                    os.remove(output_path)

    if os.path.exists(file_path):
        os.remove(file_path)
