import streamlit as st
import os
import tempfile
import subprocess
import json
from datetime import timedelta


# Helper functions
def get_file_duration(file_path: str) -> float:
    """Get duration of audio or video file using ffprobe"""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1:nokey=1",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        st.error(f"Error reading file duration: {str(e)}")
        return 0


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_codec_info(file_path: str) -> list:
    """Get codec information using ffprobe"""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0,a:0",
            "-show_entries",
            "stream=codec_type,codec_name,codec_long_name",
            "-of",
            "json",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        return data.get("streams", [])
    except Exception as e:
        st.warning(f"Could not read codec info: {str(e)}")
        return []


def trim_media(
    file_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    progress_placeholder,
):
    """Trim video or audio file using ffmpeg"""
    try:

        cmd = [
            "ffmpeg",
            "-i",
            file_path,
            "-ss",
            format_time(start_time),
            "-to",
            format_time(end_time),
            "-c",
            "copy",  # Copy codecs without re-encoding for speed
            "-y",  # Overwrite output file
            output_path,
        ]

        progress_placeholder.info("🔄 Processing with FFmpeg...")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            return True
        else:
            st.error(f"FFmpeg error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        st.error("Processing timeout - file may be too large")
        return False
    except Exception as e:
        st.error(f"Error trimming file: {str(e)}")
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
