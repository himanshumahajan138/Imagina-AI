"""Cinematic generation pipeline.

Orchestrates: script → image → video → tts → (lipsync) → merge.

All backend work goes through the service facades in `services/<modality>/`
so the tier-picker (sidebar → `st.session_state.preferred_models`) is the
single source of model selection.

These functions are still Streamlit-coupled because they update progress
placeholders and session state as they run. A later phase will split the
pure-data parts out so they can be reused outside Streamlit.
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
from pydub import AudioSegment

from core.config import COMMON_LANGUAGES, SPEAKER_OPTIONS
from core.errors import GenerationFailed, ImaginaError
from core.logger import logger
from core.registry import session_preferred
from core.worker_client import worker
from services.media.watermark import WATERMARK, logo_addition, watermark_addition
from services.video.service import VideoService  # introspection only (cfg/duration)


# ─── Helpers ────────────────────────────────────────────────────────


def hash_df(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def save_uploaded_file(uploaded_file) -> str:
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"{uuid.uuid4()}_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(temp_path)


def _materialize_reference_images(refs) -> list[str]:
    """Coerce a heterogeneous reference-image list to absolute disk paths.

    The worker reads files by path off the shared filesystem, so any
    Streamlit `UploadedFile` (in-memory bytes) needs to be persisted first.
    Already-on-disk paths or `Path` objects pass through unchanged.
    """
    out: list[str] = []
    for r in refs:
        if isinstance(r, (str, Path)):
            out.append(str(r))
        elif hasattr(r, "read"):
            tmp = Path(tempfile.gettempdir()) / f"refimg_{uuid.uuid4()}.png"
            with open(tmp, "wb") as f:
                f.write(r.read())
            if hasattr(r, "seek"):
                r.seek(0)
            out.append(str(tmp))
    return out


def _video_service() -> VideoService:
    """Build a fresh VideoService for the active session preferences.

    Cheap: just imports the chosen backend module — heavy weights load
    lazily via `core.model_manager` only on the first generation call.
    """
    return VideoService()


def video_produces_audio(_model_type: str | None = None) -> bool:
    """Return True if the active video backend natively produces audio."""
    try:
        return _video_service().produces_audio
    except ImaginaError:
        return False


def _scene_duration() -> int:
    """Per-scene clip length the active video backend emits, in seconds."""
    try:
        return _video_service().scene_duration
    except ImaginaError:
        return 10


# ─── Image generation orchestration ─────────────────────────────────


def _generate_storyboard_images(
    scene_script_pairs: List[Dict[str, str | list]],
    dimension: str,
    model_type: str | None = None,  # kept for signature compat with callers
):
    temp_dir = tempfile.mkdtemp(prefix="scene_images_")
    logger.info(f"Temporary image directory created: {temp_dir}")

    refinement_mode = bool(st.session_state.get("image_refinement_mode"))
    previous_response_id: str | None = None

    # Reference images may be UploadedFile objects (in-memory bytes) — persist
    # them to disk so the worker can read them by path.
    reference_image_paths = _materialize_reference_images(
        st.session_state.get("custom_reference_images") or []
    )

    for i, item in enumerate(scene_script_pairs):
        if item.get("custom_image"):
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
            full_path = Path(temp_dir) / f"{uuid.uuid4()}.png"
            per_scene_refs = _materialize_reference_images(item.get("reference_images") or [])
            kwargs = {
                "script": item["script"],
                "video_scene": item["video_scene"],
                "reference_text": item.get("reference_text"),
                "old_image": item.get("old_image"),
            }
            if refinement_mode:
                # response_id chaining is OpenAI-specific — other backends
                # silently ignore the kwarg, so this is safe to always pass.
                kwargs["previous_response_id"] = previous_response_id

            asset = worker.generate_image(
                prompt=item["scene"],
                out_path=full_path,
                dimension=dimension,
                reference_images=per_scene_refs or reference_image_paths,
                model_id=session_preferred("image"),
                **kwargs,
            )

            if refinement_mode and asset.meta.get("response_id"):
                previous_response_id = asset.meta["response_id"]

            yield {
                "scene_index": i,
                "scene": item["scene"],
                "script": item["script"],
                "video_scene": item["video_scene"],
                "image_path": str(asset.path),
                "custom_image_path": None,
                "image_caption": "Generated Image",
            }
        except Exception as e:
            logger.exception(f"Error generating image for scene {i+1}: {e}")


# ─── Audio: TTS + duration adjustment + BGM mixing ──────────────────


# Approx narration rate at TTS speed=1.0. English cinematic narration runs
# slightly slower than conversational speech (~150 WPM ≈ 2.5 wps); 2.4 wps
# is a safe centre that lands within the safe ±15% speed band most of the
# time, so the post-TTS fit pass barely has to do any work.
_WORDS_PER_SECOND_AT_NORMAL = 2.4
_TTS_SPEED_MIN = 0.7
_TTS_SPEED_MAX = 1.4


def _estimate_tts_speed(text: str, target_seconds: float) -> float:
    """Pre-compute a TTS speed multiplier so the synthesized audio lands
    close to `target_seconds` on the first call.

    Returns a multiplier clamped to a perceptually safe band. The post-TTS
    `fit` pass cleans up whatever residual gap remains.
    """
    words = len(text.split())
    if words == 0 or target_seconds <= 0:
        return 1.0
    natural_seconds = words / _WORDS_PER_SECOND_AT_NORMAL
    speed = natural_seconds / target_seconds
    return max(_TTS_SPEED_MIN, min(_TTS_SPEED_MAX, speed))


def adjust_audio_duration(audio_path: str, target_duration: float, method: str = "smart"):
    logger.info(
        f"[adjust] Starting | file={audio_path} | target={target_duration}s | method={method}"
    )

    audio = AudioSegment.from_wav(audio_path)
    current_duration = len(audio) / 1000.0
    duration_diff = target_duration - current_duration

    # Smart/silence/stretch/speed modes accept ±100ms as "close enough" for
    # natural-sounding output. `fit` mode insists on exact alignment so we
    # skip this early return there.
    if abs(duration_diff) < 0.1 and method != "fit":
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

    if method == "fit":
        # Simple rule: short → pad with silence at the end. Long → speed
        # up via pydub.speedup (phase-vocoder, preserves pitch). Always
        # lands EXACTLY on target_duration.
        target_ms = int(target_duration * 1000)
        current_ms = len(audio)
        diff_ms = target_ms - current_ms

        if diff_ms == 0:
            return audio_path, audio, False, 1.0

        if diff_ms > 0:
            # SHORT → silence-pad at the end.
            audio = audio + AudioSegment.silent(duration=diff_ms)
        else:
            # LONG → speed up. pydub.speedup uses overlap-add, so the
            # output length isn't always exact — top up or trim to the
            # nearest millisecond afterwards so we still land on target.
            speed_factor = current_duration / target_duration
            audio = audio.speedup(playback_speed=speed_factor)
            if len(audio) < target_ms:
                audio = audio + AudioSegment.silent(duration=target_ms - len(audio))
            elif len(audio) > target_ms:
                audio = audio[:target_ms]

        audio.export(str(output_path), format="wav")
        return str(output_path), audio, False, 1.0

    return audio_path, audio, False, 1.0


def _generate_audio(script: List[Dict], custom_bgm=None):
    logger.info("[audio-gen] Starting audio generation pipeline")
    try:
        language = st.session_state.get("language")
        lang_code = COMMON_LANGUAGES[language]
        duration = _scene_duration()
        tts_model_id = session_preferred("tts")

        audio_segments = []
        temp_files: list = []

        for index, data in enumerate(script):
            # User-supplied speed (from data_editor) overrides auto. A 1.0
            # value or missing one means "let the estimator pick" — that's
            # the common case and produces close-to-target audio in one TTS
            # round trip.
            user_speed = data.get("speed")
            if user_speed and float(user_speed) != 1.0:
                tts_speed = float(user_speed)
            else:
                tts_speed = _estimate_tts_speed(data["script"], duration)

            temp_audio_path = Path(tempfile.gettempdir()) / f"generated_{uuid.uuid4()}.wav"
            worker.synthesize(
                text=data["script"],
                out_path=temp_audio_path,
                voice=SPEAKER_OPTIONS[data["speaker"]],
                speed=tts_speed,
                language=lang_code,
                model_id=tts_model_id,
            )
            temp_files.append(temp_audio_path)

            # Single fit pass: short → silence pad, long → speed up.
            fit_path, fit_seg, _, _ = adjust_audio_duration(
                str(temp_audio_path),
                target_duration=duration,
                method="fit",
            )
            audio_segments.append(fit_seg)
            if fit_path != str(temp_audio_path):
                temp_files.append(fit_path)

            actual_s = len(fit_seg) / 1000.0
            logger.info(
                f"[audio-gen] scene {index + 1}: tts_speed={tts_speed:.2f} "
                f"target={duration}s actual={actual_s:.3f}s"
            )

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
    # Phase entry: free the LLM before we start loading TTS / image weights.
    # The LLM was kept resident through the whole script-iteration phase
    # so the user could regenerate or edit cheaply; now they've committed
    # to moving forward, drop it. No-op if the LLM was never loaded
    # (user uploaded a script directly). Best-effort.
    worker.evict_models(modality="llm")

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

    # No eviction here. SDXL/Z-Image and Kokoro stay resident while the
    # user reviews + iterates on the storyboard (regen single scene,
    # replace image, regen audio). Eviction happens at the entry to the
    # next phase (generate_video) so iteration is cheap.


# ─── Per-scene video generation ─────────────────────────────────────


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

    try:
        worker.generate_video(
            prompt=prompt,
            out_path=temp_file,
            dimension=dimension,
            duration=float(scene.get("duration") or _scene_duration()),
            seed_image=Path(scene["image_path"]) if scene.get("image_path") else None,
            model_id=session_preferred("video"),
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
    # Phase entry: free SDXL/Z-Image and Kokoro now that the user has
    # committed to video gen. They were kept resident through storyboard
    # iteration so per-scene image/audio regens were instant; eviction
    # here makes room for the (much heavier) video model. Best-effort.
    worker.evict_models(modality="image")
    worker.evict_models(modality="tts")

    if "scene_videos" not in st.session_state:
        st.session_state.scene_videos = dict()
    if "scene_video_data" not in st.session_state:
        st.session_state.scene_video_data = []

    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    video_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0, text="📑 Preparing scenes metadata...")
    status_box = status_placeholder.status("Processing...", expanded=True)

    scene_dur = _scene_duration()
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
                "duration": scene_dur,
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

    # No eviction here. The video model stays resident while the user
    # reviews + iterates on per-scene videos (regen single scene from the
    # video gallery). Eviction happens at the entry to final_generation
    # so per-scene regens are cheap.
    return True


# ─── Final merge ────────────────────────────────────────────────────


def _run_lipsync(merged_video: Path, audio_path: str) -> str | None:
    """Run the active lipsync backend; return the output path or None."""
    out_path = Path(tempfile.gettempdir()) / f"lipsync_{uuid.uuid4()}.mp4"
    try:
        asset = worker.apply_lipsync(
            video_path=merged_video,
            audio_path=Path(audio_path),
            out_path=out_path,
            model_id=session_preferred("lipsync"),
        )
        return str(asset.path)
    except ImaginaError as e:
        logger.error(f"Lipsync backend failed: {e}")
    except Exception:
        logger.exception("Lipsync crashed")
    return None


def final_generation(video_data, use_custom_audio, final_quality):
    """Merge all per-scene videos into one final output, optionally lip-synced."""
    # Phase entry: free local video weights before the merge + lipsync
    # phase. Per-scene video regens kept the model warm; now they're done.
    # No-op for API-tier video backends. Best-effort.
    worker.evict_models(modality="video")

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
                # produced an audio-synced talking head — saves a costly
                # round-trip and avoids double-syncing.
                native_audio = video_produces_audio()
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
                    lipsynced_video_path = _run_lipsync(merged_tmp, audio_path)
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
