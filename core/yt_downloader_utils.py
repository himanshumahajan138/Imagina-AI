import os
import re
import tempfile
import subprocess
from typing import Optional
from core.logger_utils import logger


def is_valid_youtube_url(url):
    """Validate YouTube URL"""
    youtube_regex = r"(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/"
    return re.match(youtube_regex, url) is not None


def extract_audio_from_hybrid(hybrid_file_path: str) -> str:
    output_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    abs_hybrid_file_path = os.path.abspath(hybrid_file_path)

    logger.info(
        f"Extracting audio from hybrid file | input={abs_hybrid_file_path}, output={output_audio_path}"
    )

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                abs_hybrid_file_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                output_audio_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Extracted audio saved at {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError:
        logger.error(f"Failed to extract audio from hybrid file: {hybrid_file_path}")
        raise ValueError(
            "Failed to extract audio from hybrid file. Please check the file and try again."
        )


def extract_video_from_hybrid(hybrid_file_path: str) -> str:
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    abs_hybrid_file_path = os.path.abspath(hybrid_file_path)

    logger.info(
        f"Extracting video from hybrid file | input={abs_hybrid_file_path}, output={output_video_path}"
    )

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                abs_hybrid_file_path,
                "-an",
                "-vcodec",
                "copy",
                output_video_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Extracted video saved at {output_video_path}")
        return output_video_path
    except subprocess.CalledProcessError:
        logger.error(f"Failed to extract video from hybrid file: {hybrid_file_path}")
        raise ValueError(
            "Failed to extract video from hybrid file. Please check the file and try again."
        )


def extract_youtube_only_audio(youtube_url: str) -> str:
    output_audio_path = tempfile.NamedTemporaryFile(
        delete=False, suffix=".%(ext)s"
    ).name
    logger.info(
        f"Extracting YouTube audio | url={youtube_url}, output={output_audio_path}"
    )
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bestaudio/best",
                "--extract-audio",
                "-o",
                output_audio_path,
                "--print",
                "after_move:filepath",
                youtube_url,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        audio_path = result.stdout.strip()
        logger.info(f"Successfully extracted audio | saved_at={audio_path}")
        return audio_path
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Failed to extract audio from YouTube | url={youtube_url}, error={e.stderr.strip()}"
        )
        raise ValueError(
            "Failed to extract audio from YouTube URL. Please check the URL and try again."
        )


def extract_youtube_only_video(youtube_url: str) -> str:
    output_video_path = tempfile.NamedTemporaryFile(
        delete=False, suffix=".%(ext)s"
    ).name
    logger.info(
        f"Extracting YouTube video (video-only) | url={youtube_url}, output={output_video_path}"
    )

    try:
        result_video = subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bestvideo",
                "-o",
                output_video_path,
                "--print",
                "after_move:filepath",
                youtube_url,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        video_path = result_video.stdout.strip()
        if video_path:
            logger.info(
                f"Successfully extracted video-only stream | saved_at={video_path}"
            )
            return video_path
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"Failed to extract only video stream | url={youtube_url}, error={e.stderr.strip()}"
        )

    logger.info(f"Falling back to hybrid stream extraction | url={youtube_url}")
    try:
        result_hybrid = subprocess.run(
            [
                "yt-dlp",
                "-f",
                "best",
                "-o",
                output_video_path,
                "--print",
                "after_move:filepath",
                youtube_url,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        hybrid_path = result_hybrid.stdout.strip()
        logger.debug(f"Hybrid stream downloaded | temp_file={hybrid_path}")
        output_video_path = extract_video_from_hybrid(hybrid_path)
        logger.info(
            f"Successfully extracted video from hybrid stream | saved_at={output_video_path}"
        )
        return output_video_path
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Failed to extract hybrid stream | url={youtube_url}, error={e.stderr.strip()}"
        )
        raise ValueError(f"Failed to extract hybrid stream: {e.stderr.strip()}")


def extract_youtube_hybrid(youtube_url: str, output_path: Optional[str] = None) -> str:
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".%(ext)s").name
    logger.info(
        f"Extracting YouTube hybrid (video+audio) | url={youtube_url}, output={output_path}"
    )

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bestvideo+bestaudio/best",
                "-o",
                output_path,
                "--print",
                "after_move:filepath",
                youtube_url,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        hybrid_path = result.stdout.strip()
        logger.info(f"Successfully extracted hybrid stream | saved_at={hybrid_path}")
        return hybrid_path
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Failed to extract hybrid stream | url={youtube_url}, error={e.stderr.strip()}"
        )
        raise ValueError(f"Failed to extract video: {e.stderr.strip()}")


def save_youtube_video_from_url(url: str, save_type: str):

    logger.info(f"Processing YouTube URL: {url}")
    if save_type == "audio":
        path = extract_youtube_only_audio(url)
        logger.info(f"Extracted YouTube audio: {path}")
        return {"audio_path": path}
    elif save_type == "video":
        path = extract_youtube_only_video(url)
        logger.info(f"Extracted YouTube video: {path}")
        return {"video_path": path}
    elif save_type == "mix":
        path = extract_youtube_hybrid(url)
        logger.info(f"Extracted YouTube hybrid: {path}")
        return {"hybrid_path": path}
    elif save_type == "both":
        video_path = extract_youtube_only_video(url)
        audio_path = extract_youtube_only_audio(url)
        logger.info(
            f"Extracted YouTube both -> video_path={video_path}, audio_path={audio_path}"
        )
        return {"video_path": video_path, "audio_path": audio_path}


def youtube_downloader_pipeline(url: str, save_type: str) -> dict:
    """
    Process YouTube URL and extract media files

    Args:
        url: YouTube URL to download
        save_type: Type of file to save ('audio', 'video', 'mix', 'both')

    Returns:
        dict: Response with 'success' flag and either 'data' or 'error'
    """
    logger.info(f"Processing request | url={url}, save_type={save_type}")

    try:
        # Validate YouTube URL
        if not is_valid_youtube_url(url):
            error_msg = "Invalid YouTube URL provided"
            logger.error(f"URL validation failed | url={url}")
            return {"success": False, "error": error_msg}

        # Call the local function to extract media
        result = save_youtube_video_from_url(url, save_type)

        logger.info(f"Successfully processed request | url={url}, result={result}")
        return {"success": True, "data": result}

    except ValueError as e:
        error_msg = f"Extraction error: {str(e)}"
        logger.error(f"ValueError during extraction | url={url}, error={e}")
        return {"success": False, "error": error_msg}

    except subprocess.CalledProcessError as e:
        error_msg = f"Process error: {str(e)}"
        logger.error(f"Subprocess error | url={url}, error={e}")
        return {"success": False, "error": error_msg}

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error in processing | url={url}, error={e}")
        return {"success": False, "error": error_msg}
