"""Cinematic generation pipeline.

Orchestrates: script → image → video → tts → (lipsync) → merge.

Functions here are heavily Streamlit-coupled because they update progress
placeholders and session state as they run. Phase 4+ will rewrite the core
generation steps to be pure (returning data) and move all UI into the tab/
component modules. For now they are moved as-is so call sites can import
them from a stable home.
"""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List

import ffmpeg
import librosa
import pandas as pd
import soundfile as sf
import streamlit as st
from google.genai import types as genai_types
from PIL import Image
from pydub import AudioSegment

from core.config import (
    COMMON_LANGUAGES,
    SPEAKER_OPTIONS,
)
from core.errors import GenerationFailed
from core.logger import logger
from core.registry import session_preferred
from services.image.backends.gemini import gemini_image_generator
from services.image.backends.openai import openai_image_generator
from services.lipsync.backends.sync_api import lipsync_generation_pipeline
from services.media.watermark import (
    WATERMARK,
    logo_addition,
    watermark_addition,
)
from services.tts.backends.kokoro_local import KokoroAudioPipeline
from services.video.service import VideoService


# Maps the user-facing "Model Type" sidebar choice to a canonical model_id
# that lives in configs/models.yaml. Keep this list short — it's the bridge
# between legacy UI labels and the new registry.
_MODEL_TYPE_TO_VIDEO_ID = {
    "openai": "sora",
    "gemini": "veo-3",
}


# ─── Small helpers used by app/ui ───────────────────────────────────


def hash_df(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def save_uploaded_file(uploaded_file) -> str:
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"{uuid.uuid4()}_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(temp_path)


# ─── Image generation orchestration ─────────────────────────────────


def _generate_storyboard_images(
    scene_script_pairs: List[Dict[str, str | list]],
    dimension: str,
    model: str,
):
    temp_dir = tempfile.mkdtemp(prefix="scene_images_")
    logger.info(f"Temporary image directory created: {temp_dir}")
    previous_response_id = None

    for i, item in enumerate(scene_script_pairs):
        if item["custom_image"]:
            image_data = item["custom_image"]
            filename = f"{uuid.uuid4()}.png"
            full_path = os.path.join(temp_dir, filename)
            with open(full_path, "wb") as f:
                f.write(base64.b64decode(image_data.split(",")[1]))
            yield {
                "scene_index": i,
                "scene": item["scene"],
                "video_scene": item["video_scene"],
                "script": item["script"],
                "image_path": None,
                "custom_image_path": full_path,
                "image_caption": "Custom Image",
            }
            continue
        try:
            filename = f"{uuid.uuid4()}.png"
            full_path = os.path.join(temp_dir, filename)
            if st.session_state.image_refinement_mode:
                full_path, response_id = openai_image_generator(
                    item=item,
                    dimension=dimension,
                    out_path=full_path,
                    previous_response_id=previous_response_id,
                )
                previous_response_id = response_id
            else:
                full_path = gemini_image_generator(item, dimension, full_path)

            if full_path:
                yield {
                    "scene_index": i,
                    "scene": item["scene"],
                    "script": item["script"],
                    "video_scene": item["video_scene"],
                    "image_path": full_path,
                    "custom_image_path": None,
                    "image_caption": "Generated Image",
                }
        except Exception as e:
            logger.exception(f"Error generating image for scene {i+1}: {e}")


# ─── Audio: TTS + duration adjustment + BGM mixing ──────────────────


def adjust_audio_duration(audio_path: str, target_duration: float, method: str = "smart"):
    logger.info(
        f"[adjust] Starting | file={audio_path} | target={target_duration}s | method={method}"
    )

    audio = AudioSegment.from_wav(audio_path)
    current_duration = len(audio) / 1000.0
    duration_diff = target_duration - current_duration

    if abs(duration_diff) < 0.1:
        return audio_path, audio, False, 1.0

    output_path = Path(tempfile.gettempdir()) / f"adjusted_audio_{uuid.uuid4()}.wav"

    if method == "smart":
        if 0 < duration_diff < 1.0:
            silence = AudioSegment.silent(duration=int(duration_diff * 1000))
            adjusted_audio = audio + silence
            adjusted_audio.export(str(output_path), format="wav")
            return str(output_path), adjusted_audio, False, 1.0

        if duration_diff < -1.8:
            suggested_speed = current_duration / target_duration
            suggested_speed = max(0.85, min(1.25, suggested_speed))
            return audio_path, audio, True, suggested_speed

        if duration_diff > 1.8:
            suggested_speed = current_duration / target_duration
            suggested_speed = max(0.75, min(1.15, suggested_speed))
            return audio_path, audio, True, suggested_speed

        # Medium diff → high-quality time stretch
        y, sr = librosa.load(audio_path, sr=None)
        stretch_factor = current_duration / target_duration
        stretch_factor = max(0.90, min(1.10, stretch_factor))
        y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor, n_fft=4096)

        from scipy.ndimage import gaussian_filter1d
        y_stretched = gaussian_filter1d(y_stretched, sigma=0.5)

        sf.write(str(output_path), y_stretched, sr)
        adjusted_audio = AudioSegment.from_wav(str(output_path))
        return str(output_path), adjusted_audio, False, 1.0

    if method == "silence":
        if duration_diff > 0:
            silence = AudioSegment.silent(duration=int(duration_diff * 1000))
            adjusted_audio = audio + silence
            adjusted_audio.export(str(output_path), format="wav")
            return str(output_path), adjusted_audio, False, 1.0
        return adjust_audio_duration(audio_path, target_duration, method="stretch")

    if method == "stretch":
        y, sr = librosa.load(audio_path, sr=None)
        stretch_factor = current_duration / target_duration
        y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
        sf.write(str(output_path), y_stretched, sr)
        adjusted_audio = AudioSegment.from_wav(str(output_path))
        return str(output_path), adjusted_audio, False, 1.0

    if method == "speed":
        speed_factor = target_duration / current_duration
        adjusted_audio = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": int(audio.frame_rate * (1 / speed_factor))},
        ).set_frame_rate(audio.frame_rate)
        adjusted_audio.export(str(output_path), format="wav")
        return str(output_path), adjusted_audio, False, 1.0

    return audio_path, audio, False, 1.0


def _generate_audio(script: List[Dict], custom_bgm=None):
    logger.info("[audio-gen] Starting audio generation pipeline")
    try:
        language = st.session_state.get("language")
        tts = KokoroAudioPipeline(lang_code=COMMON_LANGUAGES[language])

        duration = (
            8
            if st.session_state.model_type == "gemini"
            else 12 if st.session_state.model_type == "openai" else 10
        )

        audio_segments = []
        temp_files: list = []

        for index, data in enumerate(script):
            current_speed = data.get("speed", 1.0)
            max_retries = 2

            for attempt in range(max_retries):
                temp_audio_path = Path(tempfile.gettempdir()) / f"generated_{uuid.uuid4()}.wav"
                tts.text_to_audio(
                    text=data["script"],
                    voice=SPEAKER_OPTIONS[data["speaker"]],
                    speed=current_speed,
                    output_file=str(temp_audio_path),
                )

                adjusted_path, adjusted_seg, needs_regen, suggested_speed = adjust_audio_duration(
                    str(temp_audio_path),
                    target_duration=duration,
                    method="smart",
                )

                if needs_regen:
                    current_speed = suggested_speed
                    temp_files.append(temp_audio_path)
                    if attempt < max_retries - 1:
                        continue

                audio_segments.append(adjusted_seg)
                temp_files.append(temp_audio_path)
                if adjusted_path != str(temp_audio_path):
                    temp_files.append(adjusted_path)
                break

        merged_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            merged_audio += segment

        merged_audio_path = Path(tempfile.gettempdir()) / f"merged_{uuid.uuid4()}.wav"
        merged_audio.export(merged_audio_path, format="wav")

        for f in temp_files:
            try:
                Path(f).unlink()
            except Exception:
                pass

        if custom_bgm:
            mixed_audio_path = Path(tempfile.gettempdir()) / f"mixed_{uuid.uuid4()}.wav"
            if hasattr(custom_bgm, "read"):
                bgm_path = Path(tempfile.gettempdir()) / f"{custom_bgm.name}_{uuid.uuid4()}.wav"
                with open(bgm_path, "wb") as f:
                    f.write(custom_bgm.read())
            else:
                bgm_path = Path(custom_bgm)

            bgm = AudioSegment.from_wav(bgm_path) - 30
            if len(bgm) < len(merged_audio):
                bgm *= (len(merged_audio) // len(bgm)) + 1
            bgm = bgm[: len(merged_audio)]
            final_audio = bgm.overlay(merged_audio)
            final_audio.export(mixed_audio_path, format="wav")
            return {"path": str(mixed_audio_path)}

        return {"path": str(merged_audio_path)}

    except Exception as e:
        logger.exception("[audio-gen] ERROR in audio generation pipeline")
        return {"error": str(e)}


def generate_audio_images(
    global_dimension: str,
    script: List[Dict],
    model_type: str,
    use_custom_audio: bool,
):
    """Generate audio (optional) + scene images, with progress UI."""
    st.session_state.video_data = {}

    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    audio_placeholder = st.empty()
    scene_placeholder = st.empty()

    progress_bar = progress_placeholder.progress(0, text="🔄 Initializing...")
    status_box = status_placeholder.status("Processing...", expanded=True)

    temp_audio_path = ""

    if use_custom_audio:
        progress_bar.progress(10, text="🎤 Generating Audio...")
        status_box.write("🎤 Generating audio...")
        audio_result = _generate_audio(script, st.session_state.get("custom_bgm"))
        if "error" in audio_result:
            err = f"❌ Audio generation failed: {audio_result['error']}"
            st.error(err)
            status_box.write(err)
        else:
            temp_audio_path = audio_result["path"]
            progress_bar.progress(25, text="✅ Audio Generated")
            status_box.write("✅ Audio Generated")
            with audio_placeholder.expander("🎧 Preview Generated Audio"):
                st.audio(temp_audio_path)

    start_progress = 30
    progress_bar.progress(start_progress, text="🖼️ Generating Scenes...")
    total_scenes = len(script)
    image_step = 60 // max(1, total_scenes)

    if "editable_images" not in st.session_state:
        st.session_state.editable_images = []

    with scene_placeholder.expander("🎬 Preview Generated Scenes"):
        for i, new_img in enumerate(
            _generate_storyboard_images(script, global_dimension, model_type)
        ):
            st.session_state.editable_images.append(
                {
                    "image_path": new_img["image_path"],
                    "scene_number": i + 1,
                    "scene": new_img["scene"],
                    "video_scene": new_img["video_scene"],
                    "script": new_img["script"],
                    "custom_image": new_img["custom_image_path"],
                }
            )

            pct = start_progress + (i + 1) * image_step
            status_box.write(f"✅ Scene {i+1} Generated")
            progress_bar.progress(
                pct,
                text=f"Generating Scene {i+2}..." if i < total_scenes - 1 else "Finalizing...",
            )

            with st.expander(f"🎬 Scene {i+1}"):
                st.text_area("📜 Script", new_img["script"], disabled=True, key=f"script_{i}")
                st.text_area("🎥 Scene", new_img["scene"], disabled=True, key=f"scene_{i}")
                st.text_area(
                    "🎥 Video Scene",
                    new_img["video_scene"],
                    disabled=True,
                    key=f"video_scene_{i}",
                )
                st.image(
                    new_img["image_path"] or new_img["custom_image_path"],
                    caption=new_img["image_caption"],
                )

    progress_bar.progress(100, text="✅ Audio and Images Generated Successfully.")
    status_box.update(label="✅ Complete", state="complete")

    st.session_state.video_data = {
        "audio_path": str(temp_audio_path),
        "images": st.session_state.editable_images,
    }

    progress_placeholder.empty()
    status_placeholder.empty()
    audio_placeholder.empty()
    scene_placeholder.empty()


# ─── Per-scene video generation ─────────────────────────────────────


def _resolve_video_model_id(model_type: str) -> str | None:
    """Resolve which video model to use.

    Priority:
      1. The user's tier-picker override (sidebar → session_state).
      2. Legacy 'Model Type' sidebar value mapped via _MODEL_TYPE_TO_VIDEO_ID.
    """
    override = session_preferred("video")
    if override:
        return override
    return _MODEL_TYPE_TO_VIDEO_ID.get(model_type)


def video_produces_audio(model_type: str) -> bool:
    """Return True if the selected video backend natively produces audio."""
    model_id = _resolve_video_model_id(model_type)
    if not model_id:
        return False
    try:
        return VideoService(model_id=model_id).produces_audio
    except Exception:
        return False


def _generate_single_video(scene, ref_images, model_type, dimension):
    temp_file = (
        Path(tempfile.gettempdir())
        / f"scene_{scene['scene_number']}_{uuid.uuid4()}.mp4"
    )

    prompt = (
        f"Script: {scene['script']}\n"
        f"Image Scene: {scene['scene_text']}\n"
        f"Video Scene: {scene['video_scene_text']}"
    )

    model_id = _resolve_video_model_id(model_type)
    if not model_id:
        logger.warning(f"No video model_id mapped for model_type={model_type!r}")
        return ""

    try:
        VideoService(model_id=model_id).generate_video(
            prompt=prompt,
            out_path=temp_file,
            dimension=dimension,
            duration=float(scene.get("duration") or 0),
            seed_image=Path(scene["image_path"]) if scene.get("image_path") else None,
        )
    except GenerationFailed as e:
        logger.error(f"Video generation failed: {e}")
        return ""
    except Exception:
        logger.exception("Video generation crashed")
        return ""

    if temp_file.exists():
        if st.session_state.use_logo and st.session_state.custom_logo:
            uploaded_image_path = save_uploaded_file(st.session_state.custom_logo)
            temp_file = logo_addition(
                temp_file, uploaded_image_path, st.session_state.logo_location
            )

        if Path(WATERMARK).exists() and st.session_state.watermark:
            temp_file = watermark_addition(temp_file)

        return str(temp_file)
    return ""


def generate_video(video_data, global_duration, global_dimension, model_type):
    """Generate per-scene videos with progress UI; populates session state."""
    if "scene_videos" not in st.session_state:
        st.session_state.scene_videos = dict()
    if "scene_video_data" not in st.session_state:
        st.session_state.scene_video_data = []

    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    video_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0, text="📑 Preparing scenes metadata...")
    status_box = status_placeholder.status("Processing...", expanded=True)

    scenes = []
    ref_images = ""
    for i, img in enumerate(video_data["images"]):
        scenes.append(
            {
                "scene_number": img["scene_number"],
                "script": img["script"],
                "scene_text": img["scene"],
                "video_scene_text": img["video_scene"],
                "image_path": img.get("custom_image") or img["image_path"],
                "duration": (
                    8
                    if st.session_state.model_type == "gemini"
                    else 12 if st.session_state.model_type == "openai" else 10
                ),
            }
        )
        ref_images += (
            scenes[i]["image_path"] + ","
            if i < len(video_data["images"]) - 1
            else scenes[i]["image_path"]
        )

    st.session_state.scene_video_data = scenes
    scene_count = len(scenes)

    with video_placeholder.expander("🎬 Preview Generated Scene Videos"):
        for scene in scenes:
            status_box.write(f"🎬 Generating Scene {scene['scene_number']} of {scene_count}...")
            temp_file = _generate_single_video(
                scene, ref_images, model_type, global_dimension
            )
            st.session_state.scene_videos[scene["scene_number"]] = {
                "script": scene["script"],
                "scene_text": scene["scene_text"],
                "video_scene_text": scene["video_scene_text"],
                "video_path": str(temp_file),
            }
            if st.session_state.scene_videos[scene["scene_number"]]:
                with st.expander(f"Scene {scene['scene_number']}"):
                    st.text_area("Script", scene["script"], disabled=True)
                    st.text_area("Scene Text", scene["scene_text"], disabled=True)
                    st.text_area("Video Scene Text", scene["video_scene_text"], disabled=True)
                    video_path = st.session_state.scene_videos[scene["scene_number"]]["video_path"]
                    if video_path:
                        st.video(video_path)
                    else:
                        st.error("❌ Video generation failed for this scene.")

            progress_bar.progress(scene["scene_number"] / scene_count)
            status_box.write(f"✅ Scene {scene['scene_number']} Generated")

        progress_bar.progress(100, text="✅ Scene Videos Generated Successfully.")
        status_box.update(label="✅ Complete", state="complete")

    progress_placeholder.empty()
    status_placeholder.empty()
    video_placeholder.empty()


# ─── Final merge ────────────────────────────────────────────────────


def final_generation(video_data, use_custom_audio, final_quality):
    """Merge all per-scene videos into one final output, optionally lip-synced."""
    with st.status("🔗 Merging scenes into one final video...") as status:
        try:
            merged_tmp = Path(tempfile.gettempdir()) / f"merged_{uuid.uuid4()}.mp4"
            input_streams: list = []
            has_audio = False

            for scene_info in st.session_state.scene_videos.values():
                video_path = str(scene_info["video_path"])
                inp = ffmpeg.input(video_path)
                input_streams.append(inp.video)
                probe = ffmpeg.probe(video_path)
                if any(stream["codec_type"] == "audio" for stream in probe["streams"]):
                    input_streams.append(inp.audio)
                    has_audio = True

            concat = ffmpeg.concat(*input_streams, v=1, a=1 if has_audio else 0).node
            v = concat[0]
            a = concat[1] if has_audio else None

            if has_audio:
                out = ffmpeg.output(v, a, str(merged_tmp))
            else:
                out = ffmpeg.output(v, str(merged_tmp))

            out.overwrite_output().run(capture_stdout=True, capture_stderr=True)

            final_output = Path(tempfile.gettempdir()) / f"final_{uuid.uuid4()}.mp4"
            video_input = ffmpeg.input(str(merged_tmp))
            audio_attached = False

            if use_custom_audio:
                audio_path = video_data.get("audio_path")
                # Skip lip-sync when the video backend (e.g. VEO 3) already
                # produced an audio-synced talking head. Saves a costly Sync.so
                # round-trip and avoids double-syncing.
                native_audio = video_produces_audio(st.session_state.model_type)
                if (
                    st.session_state.lipsync_mode
                    and audio_path
                    and Path(audio_path).exists()
                    and not native_audio
                ):
                    status.update(
                        label="🎬 Starting lip sync (this may take a few minutes)...",
                        state="running",
                    )
                    lipsynced_video_path = lipsync_generation_pipeline(
                        video_path=str(merged_tmp), audio_path=audio_path
                    )
                    if lipsynced_video_path and Path(lipsynced_video_path).exists():
                        ffmpeg.output(
                            ffmpeg.input(lipsynced_video_path),
                            str(final_output),
                            vcodec="libx264", acodec="aac", audio_bitrate="300k",
                            pix_fmt="yuv420p", movflags="+faststart", vf=final_quality,
                        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
                        Path(lipsynced_video_path).unlink(missing_ok=True)
                        audio_attached = True
                        status.update(label="✅ Lip sync completed successfully!", state="complete")
                    else:
                        st.error("Lip sync processing failed. Using original video with audio.")
                        audio_input = ffmpeg.input(audio_path)
                        ffmpeg.output(
                            video_input.video, audio_input, str(final_output),
                            vcodec="libx264", acodec="aac", audio_bitrate="300k",
                            pix_fmt="yuv420p", movflags="+faststart", vf=final_quality,
                            **{"map": "0:v:0", "map": "1:a:0"},
                        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
                        audio_attached = True
                elif audio_path and Path(audio_path).exists():
                    status.update(label="🎵 Attaching audio...", state="running")
                    audio_input = ffmpeg.input(audio_path)
                    ffmpeg.output(
                        video_input.video, audio_input, str(final_output),
                        vcodec="libx264", acodec="aac", audio_bitrate="300k",
                        pix_fmt="yuv420p", movflags="+faststart", vf=final_quality,
                        **{"map": "0:v:0", "map": "1:a:0"},
                    ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
                    audio_attached = True
                else:
                    status.update(
                        label="⚠️ No audio found. The final video will have no sound.",
                        state="running",
                    )

            if not audio_attached:
                ffmpeg.output(
                    ffmpeg.input(str(merged_tmp)),
                    str(final_output),
                    vcodec="libx264", acodec="copy",
                    pix_fmt="yuv420p", movflags="+faststart", vf=final_quality,
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)

            st.session_state.final_output_path = str(final_output)
            st.session_state.video_generated = True

            Path(merged_tmp).unlink(missing_ok=True)
            status.update(label="✅ Final video generated successfully!", state="complete")

        except ffmpeg.Error as e:
            err_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"⚠️ FFmpeg error: {err_msg}")
            st.error("FFmpeg failed while merging videos.")
            status.update(label="❌ Video generation failed", state="error")
