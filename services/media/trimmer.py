"""Headless ffmpeg/ffprobe helpers + Streamlit-only display widgets.

The duration / codec / trim helpers above the divider are headless and
safe to call server-side (worker uses them via lipsync chunking). The
display_* helpers below are UI-only — they require a Streamlit script
context and should only be invoked from `ui/tabs/*`.
"""

import json
import os
import subprocess
import tempfile
from datetime import timedelta

import streamlit as st

from core.logger import logger


# ─── Headless helpers (server-safe) ─────────────────────────────────


def get_file_duration(file_path: str) -> float:
    """Get duration of audio or video file using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1:nokey=1",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error reading file duration: {e}")
        return 0


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_codec_info(file_path: str) -> list:
    """Get codec information using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0,a:0",
            "-show_entries", "stream=codec_type,codec_name,codec_long_name",
            "-of", "json",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        return data.get("streams", [])
    except Exception as e:
        logger.warning(f"Could not read codec info: {e}")
        return []


def trim_media(
    file_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    progress_placeholder=None,
) -> bool:
    """Trim video or audio file using ffmpeg.

    `progress_placeholder` is optional; if provided (Streamlit `st.empty()`
    handle), receives a status `info` line. Server-side callers pass None.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i", file_path,
            "-ss", format_time(start_time),
            "-to", format_time(end_time),
            "-c", "copy",
            "-y",
            output_path,
        ]

        if progress_placeholder is not None:
            progress_placeholder.info("🔄 Processing with FFmpeg...")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            return True
        logger.error(f"FFmpeg error: {result.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Processing timeout — file may be too large")
        return False
    except Exception as e:
        logger.error(f"Error trimming file: {e}")
        return False


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temp directory"""
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def display_timeline(duration: float, start: float, end: float):
    """Display visual timeline of trim range"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Start Time", format_time(start))
    with col2:
        st.metric("End Time", format_time(end))
    with col3:
        trim_duration = end - start
        st.metric("Trim Duration", format_time(trim_duration))
    with col4:
        st.metric("Total Duration", format_time(duration))


def display_file_info(file_path: str, file_name: str):
    """Display file information"""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    codec_info = get_codec_info(file_path)

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write(f"**File Name:** {file_name}")
        st.write(f"**File Size:** {file_size_mb:.2f} MB")

    with info_col2:
        if codec_info:
            for stream in codec_info:
                codec_type = stream.get("codec_type", "unknown").upper()
                codec_name = stream.get("codec_name", "unknown")
                st.write(f"**{codec_type} Codec:** {codec_name}")


def is_audio_file(file_name: str) -> bool:
    """Check if file is audio based on extension"""
    audio_extensions = (".mp3", ".wav", ".aac", ".flac", ".m4a", ".wma", ".ogg")
    return file_name.lower().endswith(audio_extensions)


def display_media_player(file_path: str, file_name: str):
    """Display audio or video player based on file type"""
    st.subheader("🎥 Media Preview")

    if is_audio_file(file_name):
        st.audio(file_path, format="audio/mpeg")
    else:
        st.video(file_path)
