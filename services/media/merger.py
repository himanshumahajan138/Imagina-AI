"""Video chunk + concat helpers (ffmpeg-based).

Headless: callable from both the Streamlit UI and the worker daemon.
Errors are logged, not surfaced via st.* — UI-layer callers can wrap
the return value if they want a banner.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from core.logger import logger
from services.media.trimmer import get_file_duration, trim_media


def split_media_into_chunks(file_path: str, max_duration: float = 299) -> list:
    """Split a media file into ≤ max_duration second chunks.

    Returns a list of (start_time, end_time, chunk_path) tuples.
    """
    total_duration = get_file_duration(file_path)
    if total_duration <= max_duration:
        return [(0, total_duration, file_path)]

    chunks: list[tuple[float, float, str]] = []
    num_chunks = int(total_duration / max_duration) + 1

    for i in range(num_chunks):
        start_time = i * max_duration
        end_time = min(start_time + max_duration, total_duration)

        file_stem = Path(file_path).stem
        file_ext = Path(file_path).suffix
        chunk_path = Path(tempfile.gettempdir()) / f"{file_stem}_chunk_{i}{file_ext}"

        if trim_media(file_path, start_time, end_time, str(chunk_path)):
            chunks.append((start_time, end_time, str(chunk_path)))

    return chunks


def merge_videos(video_chunks: list, output_path: str) -> bool:
    """Concat-demux merge a list of video file paths."""
    try:
        concat_file = Path(tempfile.gettempdir()) / "concat_list.txt"
        with open(concat_file, "w") as f:
            for chunk in video_chunks:
                f.write(f"file '{chunk}'\n")

        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy", "-y",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            concat_file.unlink()
            return True
        logger.error(f"Merge error: {result.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error merging videos: {str(e)}")
        return False
