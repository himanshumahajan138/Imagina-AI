"""Backwards-compat re-exports.

The old monolithic `core.utils` module has been split across `services/`,
`pipelines/`, and `ui/components/`. This module re-exports the public
names so any straggler imports (older scripts, notebooks, third-party
extensions) keep working. New code should import from the canonical home.

Canonical homes:
    services.media.watermark    : extract_frame, remove_watermark_*,
                                   watermark_addition, logo_addition,
                                   normalize_veo3_video, crop_image_to_dimension,
                                   WATERMARK
    services.media.merger       : merge_videos, split_media_into_chunks
    services.llm.prompts        : SCRIPT_PROMPT, IMAGE_PROMPT
    services.llm.parser         : validate_script_data, parse_script_scene_content
    services.llm.backends.openai : openai_script_generator
    services.image.backends.openai : openai_image_generator
    services.image.backends.gemini : gemini_image_generator
    services.video.backends.openai : sora_video_generation_pipeline
    services.video.backends.google_flow : gemini_video_generation_pipeline
    services.lipsync.backends.sync_api : sync_so_lipsync_pipeline,
                                          lipsync_generation_pipeline
    pipelines.cinematic         : generate_audio_images, generate_video,
                                   final_generation, hash_df, save_uploaded_file
    ui.components.storyboard_gallery : storyboard_gallery, video_gallery
"""

from __future__ import annotations

from pipelines.cinematic import (  # noqa: F401
    final_generation,
    generate_audio_images,
    generate_video,
    hash_df,
    save_uploaded_file,
)
from services.image.backends.gemini import gemini_image_generator  # noqa: F401
from services.image.backends.openai import openai_image_generator  # noqa: F401
from services.lipsync.backends.sync_api import (  # noqa: F401
    lipsync_generation_pipeline,
    sync_so_lipsync_pipeline,
)
from services.llm.backends.openai import openai_script_generator  # noqa: F401
from services.llm.parser import (  # noqa: F401
    parse_script_scene_content,
    validate_script_data,
)
from services.llm.prompts import IMAGE_PROMPT, SCRIPT_PROMPT  # noqa: F401
from services.media.merger import merge_videos, split_media_into_chunks  # noqa: F401
from services.media.watermark import (  # noqa: F401
    WATERMARK,
    crop_image_to_dimension,
    extract_frame,
    logo_addition,
    normalize_veo3_video,
    remove_watermark_ffmpeg,
    remove_watermark_opencv,
    watermark_addition,
)
from services.video.backends.google_flow import gemini_video_generation_pipeline  # noqa: F401
from services.video.backends.openai import sora_video_generation_pipeline  # noqa: F401
from ui.components.storyboard_gallery import storyboard_gallery, video_gallery  # noqa: F401

__all__ = [
    "IMAGE_PROMPT",
    "SCRIPT_PROMPT",
    "WATERMARK",
    "crop_image_to_dimension",
    "extract_frame",
    "final_generation",
    "gemini_image_generator",
    "gemini_video_generation_pipeline",
    "generate_audio_images",
    "generate_video",
    "hash_df",
    "lipsync_generation_pipeline",
    "logo_addition",
    "merge_videos",
    "normalize_veo3_video",
    "openai_image_generator",
    "openai_script_generator",
    "parse_script_scene_content",
    "remove_watermark_ffmpeg",
    "remove_watermark_opencv",
    "save_uploaded_file",
    "sora_video_generation_pipeline",
    "split_media_into_chunks",
    "storyboard_gallery",
    "sync_so_lipsync_pipeline",
    "validate_script_data",
    "video_gallery",
    "watermark_addition",
]
