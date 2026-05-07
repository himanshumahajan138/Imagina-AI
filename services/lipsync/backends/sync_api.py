"""Sync.so API lip-sync backend (chunk-aware, uses static file server)."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from sync import Sync
from sync.common import Audio, GenerationOptions, Video
from sync.core.api_error import ApiError

from core.errors import GenerationFailed
from core.logger import logger
from core.types import MediaAsset, Tier
from server.static import upload_file_to_static_server
from services.media.merger import merge_videos, split_media_into_chunks
from services.media.trimmer import format_time, get_file_duration


_client: Sync | None = None


def get_client() -> Sync:
    global _client
    if _client is None:
        _client = Sync(
            timeout=600,
            follow_redirects=True,
            api_key=os.getenv("SYNC_API_KEY"),
        )
    return _client


def sync_so_lipsync_pipeline(
    audio_url: str,
    video_url: str,
    output_dir: str | None = None,
):
    """Submit + poll a Sync.so lip-sync job; download result to a temp file."""
    if not (audio_url or video_url):
        logger.error("Please Provide Valid Video and Audio URL")
        return None

    logger.info("Starting lip sync generation job...")
    try:
        response = get_client().generations.create(
            input=[Video(url=video_url), Audio(url=audio_url)],
            model="lipsync-2-pro",
            options=GenerationOptions(sync_mode="remap"),
            output_file_name="quickstart",
        )

        if not response:
            logger.error("No Response Received during Lipsync Generation")
            return None

        job_id = response.id
        logger.info(f"Generation submitted successfully, job id: {job_id}")

        generation = get_client().generations.get(job_id)
        status = generation.status

        while status not in ["COMPLETED", "FAILED", "REJECTED"]:
            logger.info(f"Polling status for generation {job_id}")
            time.sleep(10)
            generation = get_client().generations.get(job_id)
            status = generation.status

        if status != "COMPLETED":
            logger.error(f"Generation {job_id} failed with status: {status}")
            return None

        output_url = generation.output_url
        logger.info(f"Generation {job_id} completed, output url: {output_url}")
        if not output_url:
            logger.error(f"Generation {job_id} - No Output URL in Response")
            return None

        if output_dir and os.path.exists(output_dir):
            temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4", dir=output_dir, prefix="lipsync_")
        else:
            temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4", prefix="lipsync_")

        try:
            r = requests.get(output_url, stream=True, timeout=300)
            r.raise_for_status()
            with os.fdopen(temp_fd, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Video successfully downloaded to {temp_path}")
            return temp_path
        except Exception as download_error:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Error downloading video: {download_error}")
            return None

    except ApiError as e:
        logger.error(
            f"Create generation request failed with status code {e.status_code} and error {e.body}"
        )
        return None
    except Exception as e:
        logger.error(f"Error Occurred During LipSync Generation: {str(e)}")
        return None


def lipsync_generation_pipeline(video_path: str, audio_path: str):
    """Top-level pipeline that handles >299 s files via chunking."""
    video_duration = get_file_duration(video_path)
    audio_duration = get_file_duration(audio_path)

    logger.info(
        f"Video duration: {format_time(video_duration)} | "
        f"Audio duration: {format_time(audio_duration)}"
    )

    needs_split = video_duration > 299 or audio_duration > 299

    if needs_split:
        video_chunks = split_media_into_chunks(video_path, 299)
        audio_chunks = split_media_into_chunks(audio_path, 299)
        logger.info(
            f"Split into {len(video_chunks)} video chunks and {len(audio_chunks)} audio chunks"
        )

        synced_chunks = []
        for (v_start, v_end, v_chunk), (a_start, a_end, a_chunk) in zip(
            video_chunks, audio_chunks
        ):
            video_url = upload_file_to_static_server(v_chunk)
            audio_url = upload_file_to_static_server(a_chunk)
            if not video_url or not audio_url:
                continue
            synced_chunk_path = sync_so_lipsync_pipeline(
                audio_url=audio_url,
                video_url=video_url,
                output_dir=str(Path(tempfile.gettempdir())),
            )
            if synced_chunk_path:
                synced_chunks.append(synced_chunk_path)
            else:
                return None

        if synced_chunks:
            merged_output_path = Path(tempfile.gettempdir()) / "final_merged_video.mp4"
            if merge_videos(synced_chunks, str(merged_output_path)):
                for chunk in (
                    synced_chunks
                    + [c[2] for c in video_chunks]
                    + [c[2] for c in audio_chunks]
                ):
                    try:
                        Path(chunk).unlink()
                    except Exception:
                        pass
                return merged_output_path
            return None
        return None

    video_url = upload_file_to_static_server(video_path)
    if not video_url:
        logger.error("Failed to upload video to static server")
        return None
    audio_url = upload_file_to_static_server(audio_path)
    if not audio_url:
        logger.error("Failed to upload audio to static server")
        return None

    return sync_so_lipsync_pipeline(
        audio_url=audio_url,
        video_url=video_url,
        output_dir=str(Path(tempfile.gettempdir())),
    )


class SyncAPILipsyncBackend:
    name = "sync_api"
    tier = Tier.API

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def apply(
        self,
        video_path: Path,
        audio_path: Path,
        out_path: Path,
        **kwargs: Any,
    ) -> MediaAsset:
        result = lipsync_generation_pipeline(str(video_path), str(audio_path))
        if not result:
            raise GenerationFailed("Sync.so lip-sync returned no output")
        return MediaAsset(path=Path(result), kind="video")


def build_backend(cfg: dict[str, Any]) -> SyncAPILipsyncBackend:
    return SyncAPILipsyncBackend(cfg)
