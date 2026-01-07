import re
import os
import cv2
import time
import uuid
import ffmpeg
import base64
import hashlib
import requests
import tempfile
import subprocess
import librosa
import json_repair
import numpy as np
import pandas as pd
from sync import Sync
from PIL import Image
import streamlit as st
import soundfile as sf
from io import BytesIO
from google import genai
from pathlib import Path
from openai import OpenAI
from typing import List, Dict
from pydub import AudioSegment
from google.genai import types
from dotenv import load_dotenv
from core.logger_utils import logger
from sync.core.api_error import ApiError
from datetime import datetime, timedelta
from core.kokoro_tts_utils import KokoroAudioPipeline
from sync.common import Audio, GenerationOptions, Video
# from core.fastwan_utils import fastwan_video_generation
from core.static_file_serve_api import upload_file_to_static_server
from core.trimmer_utils import get_file_duration, format_time, trim_media
from core.config import ASPECT_RATIOS, FASTWAN_DIMENSIONS, SORA_DIMENSIONS, SPEAKER_OPTIONS, COMMON_LANGUAGES

# Load environment variables
load_dotenv()


WATERMARK = "images\\watermark.png"
OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GEMINI_CLIENT = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
SYNC_SO_CLIENT = Sync(
    timeout=600, follow_redirects=True, api_key=os.getenv("SYNC_API_KEY")
)
SCRIPT_PROMPT = """
You are an expert cinematic video‐script and media‐asset generator.  
Your mission is to transform a simple theme, duration, and language into a tightly‑paced, visually rich, and emotionally engaging short video blueprint—ready for automated TTS, image generation, and cinematic video production.

When I provide you with:
    • Theme: {theme}   # the central topic or concept of the video
    • Duration: {duration}  # total video length in seconds
    • Language: {language}  # the language in which the "script" lines should be written (all other prompts remain in English)

You must output **only** a JSON array of “script beats,” where each beat is an object containing exactly these five keys:

[
    {{
        "script":       "",  # One line of dialogue in {language}, ~{seconds} seconds when read aloud by TTS, written with cinematic tone
        "scene":        "",  # A vivid paragraph image prompt (in English)—cinematic mood, lighting, composition, and emotional continuity
        "video_scene":  "",  # An extended prompt (in English) directing a video generator: camera moves, motion style, real‑world physics, mood effects
        "start_time":   "",  # Beat start in HH:MM:SS,mmm never bigger than the end time
        "end_time":     ""   # Beat end in   HH:MM:SS,mmm and the last timestamp should match exactly to the duration
    }},
    …
]

**Guidelines (follow to the letter):**  
0. **Language enforcement:** All `"script"` lines must be written in the specified {language}. Both `"scene"` and `"video_scene"` prompts must always remain in English.  
1. **Compute beats:** Divide {duration} seconds into contiguous {seconds}‑second intervals (⌊{duration} / {seconds}⌋ beats).  
2. **Script lines:** Each `"script"` must be naturally speakable in ~{seconds}s, cinematic in language, and reflective of the evolving theme.  
3. **Timestamps:** Accurately calculate `"start_time"` and `"end_time"` (HH:MM:SS,mmm). No timestamp may exceed the total {duration}.  
4. **Cinematic “scene” prompts:** For each beat, craft a brief paragraph describing exactly what to image‑generate—consider composition, lighting, color palette, and emotional tone—while preserving narrative flow between beats.  
5. **Cinematic “video_scene” prompts:** For each beat, write a detailed directive for animating the image: specify camera movements (dolly, pan, tilt), motion guidance (slow‑mo, tracking), environmental effects (dust, light rays), and any mood‑enhancing filters to achieve a realistic, cinematic result.  
6. **Strict JSON only:** Return **only** the JSON array. No commentary, no extra keys, no markdown—just valid JSON.

Begin by determining the number of beats, then output the array of objects accordingly.
"""
IMAGE_PROMPT = """
You are generating a cinematic image for a short video.

Here is the current script line:
"{script}"

Here is the scene description to be illustrated:
"{scene}"

Here is the extended video direction for this beat:
"{video_scene}"

**IMPORTANT**: Use the Refrence images (If Provided) where its required to maintain a proper context

Your task:1
- Create a **visually detailed, cinematic image prompt** suitable for an image generation model.
- Use the **scene** as the foundation — bring it vividly to life.
- Incorporate the **mood, tone, and narrative feel** implied by the script line.
- Include emotional expression, lighting, composition, environment, and camera framing.
- Assume this is the **next shot in a film**, and visual continuity with the previous image must be maintained (do not reset the style or tone unless the script calls for a shift).

The output should be a **single cinematic image prompt** in English, without referencing the input keys.

Make the viewer feel like they are watching the next shot of a beautifully directed film.
"""


SCENE_COUNT = 0
GAP = 0


def extract_frame(video_path, frame_number=0):
    """Extract a frame from video for preview"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def remove_watermark_opencv(
    input_path, output_path, x, y, width, height, method="telea"
):
    """Remove watermark using OpenCV inpainting - More natural results with audio preserved"""
    # Step 1: Extract audio from original video
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".aac").name
    extract_audio_cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "copy",
        audio_path,
        "-y",
    ]
    subprocess.run(extract_audio_cmd, capture_output=True)

    # Step 2: Process video frames with OpenCV
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Temporary video without audio
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (frame_width, frame_height))

    # Create mask
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    mask[y : y + height, x : x + width] = 255

    # Choose inpainting method
    inpaint_method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inpainted = cv2.inpaint(frame, mask, 7, inpaint_method)
        out.write(inpainted)

    cap.release()
    out.release()

    # Step 3: Merge video with original audio
    merge_cmd = [
        "ffmpeg",
        "-i",
        temp_video,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-strict",
        "experimental",
        "-shortest",  # Match duration to shortest stream
        output_path,
        "-y",
    ]
    subprocess.run(merge_cmd, check=True, capture_output=True)

    # Cleanup temporary files
    if os.path.exists(temp_video):
        os.unlink(temp_video)
    if os.path.exists(audio_path):
        os.unlink(audio_path)


def remove_watermark_ffmpeg(input_path, output_path, x, y, width, height):
    """Remove watermark using FFmpeg - Faster but more blur"""
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-vf",
        f"delogo=x={x}:y={y}:w={width}:h={height}",
        "-c:a",
        "copy",
        output_path,
        "-y",
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def hash_df(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def crop_image_to_dimension(image_path: str, target_dimension: str) -> str:
    """
    Crop image to target dimension using center crop.

    Args:
        image_path: Path to input image
        target_dimension: Target dimension string (e.g., "720x1280" or "1280x720")

    Returns:
        str: Path to cropped image
    """
    target_width, target_height = map(int, target_dimension.split("x"))
    target_aspect = target_width / target_height

    logger.info(f"Cropping image to {target_dimension}")

    img = Image.open(image_path)
    img_width, img_height = img.size
    img_aspect = img_width / img_height

    logger.info(f"Original image size: {img_width}x{img_height}")

    # Check if image already matches target dimension
    if img_width == target_width and img_height == target_height:
        logger.info("Image already matches target dimension")
        return image_path

    # Calculate crop dimensions (center crop)
    if img_aspect > target_aspect:
        # Image is wider, crop width
        new_width = int(img_height * target_aspect)
        new_height = img_height
        left = (img_width - new_width) // 2
        top = 0
        right = left + new_width
        bottom = img_height
    else:
        # Image is taller, crop height
        new_width = img_width
        new_height = int(img_width / target_aspect)
        left = 0
        top = (img_height - new_height) // 2
        right = img_width
        bottom = top + new_height

    # Crop and resize to exact target dimensions
    cropped_img = img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((target_width, target_height), Image.LANCZOS)

    # Save cropped image
    temp_fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="cropped_")
    os.close(temp_fd)
    resized_img.save(temp_path)

    logger.info(f"Image cropped and saved: {temp_path}")

    return temp_path


def sora_video_generation_pipeline(
    image_path: str, prompt: str, dimension: str, duration: int, output_path: str = None
):

    logger.info(f"Starting video generation: {prompt[:50]}...")

    # Crop image to required dimension
    cropped_image_path = crop_image_to_dimension(image_path, dimension)

    video = OPENAI_CLIENT.videos.create(
        model="sora-2",
        prompt=prompt,
        size=dimension,
        input_reference=Path(cropped_image_path),
        seconds=str(duration),
        timeout=600,
    )

    logger.info(f"Video ID: {video.id} - Status: {video.status}")

    while video.status in ("in_progress", "queued"):
        logger.info(f"Polling status for generation {video.id}")
        video = OPENAI_CLIENT.videos.retrieve(video.id)
        logger.info(f"Video ID: {video.id} - Status: {video.status}")
        time.sleep(5)

    if video.status == "failed":
        logger.error(f"Sora Video generation failed: {video.id}; Reason: {video.error}")
        raise Exception(
            f"Sora Video generation failed: {video.id}; Reason: {video.error}"
        )

    logger.info("Sora Video generation completed")

    content = OPENAI_CLIENT.videos.download_content(video.id, variant="video")

    if not output_path:
        temp_fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="sora_")
        os.close(temp_fd)

    content.write_to_file(output_path)

    logger.info(f"Video saved: {output_path}")

    # Clean up temporary cropped image if it was created
    if cropped_image_path != image_path:
        try:
            os.remove(cropped_image_path)
            logger.info("Temporary cropped image cleaned up")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary image: {e}")

    return output_path


def _is_valid_srt_timestamp(ts: str) -> bool:
    pattern = r"^\d{2}:\d{2}:\d{2},\d{3}$"
    return bool(re.match(pattern, ts))


def _parse_srt_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%H:%M:%S,%f")


def _to_seconds(dt: datetime) -> float:
    """Convert a datetime object to total seconds."""
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000


def validate_script_data(script_data: list[dict], global_duration: int) -> list[str]:
    """
    Validates each row of the script_data. Returns a list of error strings.

    Checks:
    - start_time and end_time are in correct format
    - start_time < end_time
    - speaker, speed, script, scene are valid
    - each row's end_time must exactly match the next row's start_time (no gaps)
    - overall end_time does not exceed global_duration (in seconds)
    """
    errors = []
    max_end_seconds = 0

    for i, row in enumerate(script_data, start=1):
        start_time = row.get("start_time", "").strip()
        end_time = row.get("end_time", "").strip()

        # Validate timestamp format
        if not _is_valid_srt_timestamp(start_time):
            errors.append(
                f"Scene {i}: Invalid start_time format: '{start_time}'  \nPlease use format HH:MM:SS,mmm (e.g., 00:00:05,000)"
            )
        if not _is_valid_srt_timestamp(end_time):
            errors.append(
                f"Scene {i}: Invalid end_time format: '{end_time}'  \nPlease use format HH:MM:SS,mmm (e.g., 00:00:10,450)"
            )

        # Validate time logic
        if _is_valid_srt_timestamp(start_time) and _is_valid_srt_timestamp(end_time):
            start_dt = _parse_srt_timestamp(start_time)
            end_dt = _parse_srt_timestamp(end_time)
            if start_dt >= end_dt:
                errors.append(f"Scene {i}: start_time must be before end_time.")
            max_end_seconds = max(max_end_seconds, _to_seconds(end_dt))

            # Continuity check with previous row
            if i > 1:
                prev_end = _parse_srt_timestamp(script_data[i - 2]["end_time"])
                if prev_end != start_dt:
                    errors.append(
                        f"Scene {i-1} end_time `{script_data[i-2]['end_time']}` "
                        f"must exactly match Scene {i} start_time `{start_time}` (no gaps or overlaps)."
                    )

        # Field validations
        if not row.get("script") or not str(row["script"]).strip():
            errors.append(f"Scene {i}: Script cannot be empty.")
        if not row.get("scene") or not str(row["scene"]).strip():
            errors.append(f"Scene {i}: Scene cannot be empty.")
        if not row.get("video_scene") or not str(row["video_scene"]).strip():
            errors.append(f"Scene {i}: Video Scene cannot be empty.")

    # Final check for max duration
    if max_end_seconds > global_duration:
        formatted_duration = str(timedelta(seconds=global_duration))
        formatted_max = str(timedelta(seconds=max_end_seconds))
        errors.append(
            f"⏱️ Total duration exceeds global limit.  \n"
            f"- Max end_time in script: `{formatted_max}`  \n"
            f"- Allowed global duration: `{formatted_duration}`"
        )

    return errors


def save_uploaded_file(uploaded_file):
    """Save uploaded file to a unique temp path and return the path."""
    temp_dir = Path(tempfile.gettempdir())
    unique_name = f"{uuid.uuid4()}_{uploaded_file.name}"
    temp_path = temp_dir / unique_name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(temp_path)


def parse_script_scene_content(text: str) -> list[dict]:
    """
    Parses a block-structured script with SRT-style timestamps and case-insensitive [script]/[scene]/[video_scene] tags.

    Args:
        text (str): Full multiline script content.

    Returns:
        List[Dict[str, str]]: List with start_time, end_time, script, scene, and video_scene keys.
    """
    with st.spinner("Parsing and Loading Custom Script..."):
        blocks = re.split(r"\n\s*\n", text.strip())
        parsed = []

        for i, block in enumerate(blocks):
            lines = block.strip().splitlines()
            if not lines:
                continue

            # ---------- Timestamp Parsing ----------
            timestamp_line = lines[0].strip()
            timestamp_match = re.match(
                r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})",
                timestamp_line,
            )
            if not timestamp_match:
                logger.warning(
                    f"Invalid or missing timestamp in block #{i+1}. Skipping."
                )
                continue

            start_time = timestamp_match.group("start")
            end_time = timestamp_match.group("end")

            # ---------- Script, Scene, and Video Scene Parsing ----------
            script = ""
            scene = ""
            video_scene = ""

            for line in lines[1:]:
                line = line.strip()

                if line.lower().startswith("[script]:"):
                    script_match = re.match(
                        r'\[script\]:\s*"(.*?)"\s*$', line, re.IGNORECASE
                    )
                    if script_match:
                        script = script_match.group(1).strip()
                    else:
                        script = line.split(":", 1)[1].strip().strip('"')

                elif line.lower().startswith("[scene]:"):
                    scene_match = re.match(
                        r'\[scene\]:\s*"(.*?)"\s*$', line, re.IGNORECASE
                    )
                    if scene_match:
                        scene = scene_match.group(1).strip()
                    else:
                        scene = line.split(":", 1)[1].strip().strip('"')

                elif line.lower().startswith("[video_scene]:"):
                    video_scene_match = re.match(
                        r'\[video_scene\]:\s*"(.*?)"\s*$', line, re.IGNORECASE
                    )
                    if video_scene_match:
                        video_scene = video_scene_match.group(1).strip()
                    else:
                        video_scene = line.split(":", 1)[1].strip().strip('"')

            if not script:
                logger.warning(f"Missing [script] in block #{i+1}. Skipping.")
                continue
            if not scene:
                logger.warning(f"Missing [scene] in block #{i+1}. Skipping.")
                continue
            if not video_scene:
                logger.info(
                    f"No [video_scene] provided in block #{i+1}. Leaving blank."
                )

            parsed.append(
                {
                    "script": script,
                    "scene": scene,
                    "video_scene": video_scene,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

        logger.info(f"Parsed {len(parsed)} valid script-scene blocks.")
        return parsed


def openai_script_generator(
    theme: str,
    language: str,
    duration: int,
    model: str = "gpt-4o-mini",
    model_type: str = "openai",
) -> list[dict]:
    with st.spinner("Generating cinematic script..."):
        seconds = (
            8
            if model_type == "gemini"
            else 12 if st.session_state.model_type == "openai" else 10
        )

        prompt = SCRIPT_PROMPT.format(
            theme=theme, language=language, duration=duration, seconds=seconds
        )
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            raw_output = response.choices[0].message.content.strip()
            structured = json_repair.loads(raw_output)
            assert isinstance(structured, list), "Expected a list of dictionaries"
            for entry in structured:
                assert "script" in entry and "scene" in entry
            return structured
        except Exception as e:
            logger.error("OpenAI script generation failed.")
            raise e


def openai_image_generator(item, dimension, out_path, previous_response_id=None):
    # Base input with text prompt
    custom_regenrate_text = (
        ""
        if not item.get("reference_text")
        else f"\n** IMPORTANT CUSTOM INSTRUCTIONS TO BE FOLLOWED FOR IMAGE GENERATION {item.get('reference_text')}"
    )
    content = [
        {
            "type": "input_text",
            "text": IMAGE_PROMPT.format(
                scene=item["scene"],
                script=item["script"],
                video_scene=item["video_scene"],
            )
            + custom_regenrate_text,
        }
    ]

    # Append reference images if any
    custom_images_list = (
        item.get("reference_images", [])
        if item.get("reference_images", [])
        else st.session_state.get("custom_reference_images", [])
    )
    for img_file in custom_images_list:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base64_image}",
            }
        )
        img_file.seek(0)  # reset pointer

    if item.get("old_image"):
        old_image_path = item["old_image"]

        if os.path.exists(old_image_path):
            # Read and encode the old image
            with open(old_image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{base64_image}",
                }
            )
        else:
            logger.warning(f"Old image not found at: {old_image_path}")

    response = OPENAI_CLIENT.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": content}],
        tools=[
            {
                "type": "image_generation",
                "background": "opaque",
                "quality": "high" if custom_images_list else "medium",
                "size": dimension,
            }
        ],
        previous_response_id=previous_response_id or None,
    )

    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]

    if image_data:
        image_base64 = image_data[0]
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(image_base64))

    return out_path, response.id


def gemini_image_generator(item, dimension, out_path):
    dimension = ASPECT_RATIOS[dimension]
    custom_regenrate_text = (
        ""
        if not item.get("reference_text")
        else f"\n** IMPORTANT CUSTOM INSTRUCTIONS TO BE FOLLOWED FOR IMAGE GENERATION {item.get('reference_text')}"
    )
    prompt = (
        IMAGE_PROMPT.format(
            scene=item["scene"],
            script=item["script"],
            video_scene=item["video_scene"],
        )
        + custom_regenrate_text
    )

    def send_request():
        return GEMINI_CLIENT.models.generate_images(
            model="imagen-4.0-generate-preview-06-06",
            prompt=prompt,
            config=types.GenerateImagesConfig(
                output_mime_type="image/png",
                aspect_ratio=dimension,
                number_of_images=1,
            ),
        )

    response = None
    for attempt in range(3):
        try:
            logger.info(f"Attempt {attempt + 1} to generate image for scene")
            response = send_request()
            if response and response.generated_images:
                for generated_image in response.generated_images:
                    generated_image.image.save(out_path)
                logger.info(f"Image successfully generated at {out_path}")
                return out_path
            else:
                logger.warning(f"No images generated on attempt {attempt + 1}")
        except Exception as e:
            logger.error(f"Error generating image on attempt {attempt + 1}: {e}")
    logger.error(f"Failed to generate image after 3 attempts for scene")
    return None


def _generate_storyboard_images(
    scene_script_pairs: List[Dict[str, str | list]], dimension: str, model: str
):
    temp_dir = tempfile.mkdtemp(prefix="scene_images_")
    logger.info(f"Temporary image directory created: {temp_dir}")
    previous_response_id = None

    for i, item in enumerate(scene_script_pairs):
        # If a custom image is provided, use it directly
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
            image_data = None
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


def adjust_audio_duration(audio_path: str, target_duration: float, method: str = "smart"):
    logger.info(f"[adjust] Starting duration adjustment | file={audio_path} | target={target_duration}s | method={method}")

    audio = AudioSegment.from_wav(audio_path)
    current_duration = len(audio) / 1000.0
    duration_diff = target_duration - current_duration

    logger.debug(f"[adjust] Current duration={current_duration:.3f}s | Diff={duration_diff:.3f}s")

    if abs(duration_diff) < 0.1:
        logger.info("[adjust] Duration within ±0.1s → Returning original file (no adjustment needed)")
        return audio_path, audio, False, 1.0

    output_path = Path(tempfile.gettempdir()) / f"adjusted_audio_{uuid.uuid4()}.wav"

    # SMART STRATEGY
    if method == "smart":
        logger.info("[adjust] Using SMART method")

        # 1. Add silence (best quality)
        if 0 < duration_diff < 1.0:
            logger.info(f"[adjust] Small positive diff ({duration_diff:.3f}s) → Adding silence")
            silence = AudioSegment.silent(duration=int(duration_diff * 1000))
            adjusted_audio = audio + silence
            adjusted_audio.export(str(output_path), format="wav")
            logger.info(f"[adjust] Silence added successfully → {output_path}")
            return str(output_path), adjusted_audio, False, 1.0

        # 2. Audio is too long → needs speed-up regen
        elif duration_diff < -1.8:
            suggested_speed = current_duration / target_duration
            suggested_speed = max(0.85, min(1.25, suggested_speed))
            logger.warning(
                f"[adjust] Large negative diff ({duration_diff:.3f}s) → Needs regeneration with faster speed={suggested_speed:.3f}"
            )
            return audio_path, audio, True, suggested_speed

        # 3. Audio too short → needs slow-down regen
        elif duration_diff > 1.8:
            suggested_speed = current_duration / target_duration
            suggested_speed = max(0.75, min(1.15, suggested_speed))
            logger.warning(
                f"[adjust] Large positive diff ({duration_diff:.3f}s) → Needs regeneration with slower speed={suggested_speed:.3f}"
            )
            return audio_path, audio, True, suggested_speed

        # 4. Medium diff → high-quality time stretch
        else:
            logger.info(
                f"[adjust] Medium diff ({duration_diff:.3f}s) → Using high-quality time-stretching"
            )
            y, sr = librosa.load(audio_path, sr=None)
            stretch_factor = current_duration / target_duration
            stretch_factor = max(0.90, min(1.10, stretch_factor))

            logger.debug(f"[adjust] Stretch factor={stretch_factor:.3f}")

            y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor, n_fft=4096)

            from scipy.ndimage import gaussian_filter1d
            y_stretched = gaussian_filter1d(y_stretched, sigma=0.5)

            sf.write(str(output_path), y_stretched, sr)
            logger.info(f"[adjust] Time-stretch completed → {output_path}")

            adjusted_audio = AudioSegment.from_wav(str(output_path))
            return str(output_path), adjusted_audio, False, 1.0

    # METHOD: silence
    elif method == "silence":
        logger.info("[adjust] Using SILENCE method")
        if duration_diff > 0:
            logger.info(f"[adjust] Adding silence ({duration_diff:.3f}s)")
            silence = AudioSegment.silent(duration=int(duration_diff * 1000))
            adjusted_audio = audio + silence
            adjusted_audio.export(str(output_path), format="wav")
            logger.info(f"[adjust] Silence method completed → {output_path}")
            return str(output_path), adjusted_audio, False, 1.0
        else:
            logger.warning("[adjust] Negative diff → falling back to stretch method")
            return adjust_audio_duration(audio_path, target_duration, method="stretch")

    # METHOD: stretch
    elif method == "stretch":
        logger.info("[adjust] Using STRETCH method")
        y, sr = librosa.load(audio_path, sr=None)
        stretch_factor = current_duration / target_duration
        logger.debug(f"[adjust] Stretch factor={stretch_factor:.3f}")
        y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
        sf.write(str(output_path), y_stretched, sr)
        logger.info(f"[adjust] Stretch method completed → {output_path}")
        adjusted_audio = AudioSegment.from_wav(str(output_path))
        return str(output_path), adjusted_audio, False, 1.0

    # METHOD: speed
    elif method == "speed":
        speed_factor = target_duration / current_duration
        logger.info(f"[adjust] Using SPEED method | speed_factor={speed_factor:.3f}")
        adjusted_audio = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": int(audio.frame_rate * (1 / speed_factor))}
        ).set_frame_rate(audio.frame_rate)

        adjusted_audio.export(str(output_path), format="wav")
        logger.info(f"[adjust] Speed adjustment completed → {output_path}")
        return str(output_path), adjusted_audio, False, 1.0


def _generate_audio(script: List[Dict], custom_bgm=None):
    logger.info("[audio-gen] Starting audio generation pipeline")

    try:
        language = st.session_state.get("language")
        logger.info(f"[audio-gen] Selected language={language}")

        tts = KokoroAudioPipeline(lang_code=COMMON_LANGUAGES[language])

        duration = (
            8 if st.session_state.model_type == "gemini"
            else 12 if st.session_state.model_type == "openai"
            else 10
        )

        logger.info(f"[audio-gen] Target duration per segment={duration}s")

        audio_segments = []
        temp_files = []

        # GENERATE SEGMENTS
        for index, data in enumerate(script):
            logger.info(f"[audio-gen] --- Segment {index + 1}/{len(script)} ---")
            current_speed = data.get("speed", 1.0)
            max_retries = 2

            for attempt in range(max_retries):
                logger.info(f"[audio-gen] Generating audio | attempt={attempt+1} | speed={current_speed:.3f}")

                temp_audio_path = Path(tempfile.gettempdir()) / f"generated_{uuid.uuid4()}.wav"

                tts.text_to_audio(
                    text=data["script"],
                    voice=SPEAKER_OPTIONS[data["speaker"]],
                    speed=current_speed,
                    output_file=str(temp_audio_path),
                )

                logger.info("[audio-gen] Raw TTS audio generated")

                # ADJUST DURATION
                adjusted_path, adjusted_seg, needs_regen, suggested_speed = adjust_audio_duration(
                    str(temp_audio_path),
                    target_duration=duration,
                    method="smart"
                )

                if needs_regen:
                    logger.warning(
                        f"[audio-gen] Duration way off → Regeneration needed | suggested_speed={suggested_speed:.3f}"
                    )
                    current_speed = suggested_speed
                    temp_files.append(temp_audio_path)

                    if attempt < max_retries - 1:
                        continue  # retry
                else:
                    logger.info("[audio-gen] Duration OK → accepting segment")

                audio_segments.append(adjusted_seg)
                temp_files.append(temp_audio_path)

                if adjusted_path != str(temp_audio_path):
                    temp_files.append(adjusted_path)

                break  # stop retry loop

        # MERGE AUDIO
        logger.info("[audio-gen] Merging all segments...")
        merged_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            merged_audio += segment

        merged_audio_path = Path(tempfile.gettempdir()) / f"merged_{uuid.uuid4()}.wav"
        merged_audio.export(merged_audio_path, format="wav")
        logger.info(f"[audio-gen] Merged audio ready → {merged_audio_path}")

        # CLEANUP
        for f in temp_files:
            try:
                Path(f).unlink()
            except:
                logger.debug(f"[cleanup] Failed to remove temp file: {f}")

        # ADD BGM
        if custom_bgm:
            logger.info("[audio-gen] Mixing background music")
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

            logger.info(f"[audio-gen] Final mixed audio ready → {mixed_audio_path}")

            return {"path": str(mixed_audio_path)}

        logger.info(f"[audio-gen] Final audio ready → {merged_audio_path}")
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
    st.session_state.video_data = {}

    # --- UI Placeholders ---
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    audio_placeholder = st.empty()
    scene_placeholder = st.empty()

    progress_bar = progress_placeholder.progress(0, text="🔄 Initializing...")
    status_box = status_placeholder.status("Processing...", expanded=True)

    temp_audio_path = ""
    
    # --- Step 1: Generate Audio First (if needed) ---
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

    # --- Step 2: Generate Images ---
    start_progress = 30
    progress_bar.progress(start_progress, text="🖼️ Generating Scenes...")
    total_scenes = len(script)
    image_step = 60 // max(1, total_scenes)  # 30–90%

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

            # update progress
            pct = start_progress + (i + 1) * image_step
            status_box.write(f"✅ Scene {i+1} Generated")
            progress_bar.progress(pct, text=f"Generating Scene {i+2}..." if i < total_scenes - 1 else "Finalizing...")

            # scene preview
            with st.expander(f"🎬 Scene {i+1}"):
                st.text_area(
                    "📜 Script", new_img["script"], disabled=True, key=f"script_{i}"
                )
                st.text_area(
                    "🎥 Scene", new_img["scene"], disabled=True, key=f"scene_{i}"
                )
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

    # --- Step 3: Wrap up ---
    progress_bar.progress(100, text="✅ Audio and Images Generated Successfully.")
    status_box.update(label="✅ Complete", state="complete")

    st.session_state.video_data = {
        "audio_path": str(temp_audio_path),
        "images": st.session_state.editable_images,
    }

    # cleanup placeholders
    progress_placeholder.empty()
    status_placeholder.empty()
    audio_placeholder.empty()
    scene_placeholder.empty()


def storyboard_gallery(
    global_dimension: str,
    script: List[Dict],
    model_type: str,
    use_custom_audio: bool,
):
    with st.expander("🖼️ Storyboard Gallery", expanded=True):
        if use_custom_audio:
            with st.expander("🎧 Generated Audio"):
                # If preview exists, show it instead of original
                if "new_audio_data" in st.session_state:
                    st.write("Original Audio")
                    audio_path = st.session_state.video_data.get("audio_path", None)
                    if audio_path:
                        st.audio(audio_path)
                    st.divider()
                    st.write("New Generated Audio")
                    new_audio_data = st.session_state.new_audio_data
                    st.audio(new_audio_data["path"])
                    st.divider()
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button(
                            "REPLACE AUDIO",
                            key="replace_audio",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.video_data["audio_path"] = new_audio_data[
                                "path"
                            ]
                            st.session_state.pop("new_audio_data", None)
                            st.rerun()

                    with col2:
                        if st.button(
                            "CANCEL",
                            key="cancel_audio",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.pop("new_audio_data", None)
                            st.rerun()
                else:
                    # Original audio
                    audio_path = st.session_state.video_data.get("audio_path", None)
                    if audio_path:
                        st.audio(audio_path)

                    if st.button(
                        "REGENERATE AUDIO",
                        key="regen_audio",
                        width="stretch",
                        disabled=st.session_state.generating,
                    ):

                        with st.spinner("Regenerating audio..."):
                            try:
                                new_audio_data = _generate_audio(
                                    script, st.session_state.get("custom_bgm")
                                )
                                st.session_state.new_audio_data = new_audio_data
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to Regenerate Audio: {e}")

        if "new_image_data" not in st.session_state:
            st.session_state.new_image_data = {}

        for i, entry in enumerate(st.session_state.video_data["images"]):
            with st.expander(f"🎬 Scene {i + 1}"):
                st.text_area("📜 Script", value=entry["script"], disabled=True)
                st.text_area(
                    "🎥 Scene Description", value=entry["scene"], disabled=True
                )
                st.text_area(
                    "🎥 Video Scene Description",
                    value=entry["video_scene"],
                    disabled=True,
                )

                if entry.get("custom_image"):
                    st.image(
                        entry["custom_image"],
                        caption="Custom Image",
                        width="stretch",
                    )
                else:
                    st.image(
                        entry["image_path"],
                        caption="Generated Image",
                        width="stretch",
                    )

                with st.expander("REPLACE IMAGE"):
                    uploaded = st.file_uploader(
                        f"Replace Scene {i + 1} Image",
                        key=f"upload_{i}",
                        type=["png", "jpg", "jpeg"],
                        disabled=st.session_state.generating,
                    )

                    if uploaded:
                        with st.expander("Preview Uploaded Image"):
                            st.image(
                                uploaded,
                                caption="Uploaded Image",
                                width="stretch",
                            )

                        if st.button(
                            f"Update Scene {i + 1} with This Image",
                            key=f"confirm_upload_{i}",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            temp_image_path = (
                                Path(tempfile.gettempdir())
                                / f"updated_image_{uuid.uuid4()}.png"
                            )
                            img_data = uploaded.read()
                            encoded_img = base64.b64encode(img_data).decode("utf-8")
                            image_data_url = f"data:image/png;base64,{encoded_img}"
                            image_data = img_data
                            image = Image.open(BytesIO(image_data))
                            image.save(temp_image_path)

                            st.session_state.video_data["images"][i]["custom_image"] = (
                                str(temp_image_path)
                            )

                            if i < len(st.session_state.script_df):
                                st.session_state.script_df.at[i, "custom_image"] = (
                                    image_data_url
                                )

                            st.session_state.rerun_needed = True
                            st.rerun()

                # New logic: Image regeneration preview flow
                if i in st.session_state.new_image_data:
                    st.divider()
                    st.write("New Generated Image:")
                    st.image(
                        st.session_state.new_image_data[i]["image_path"],
                        caption="New Generated Image",
                        width="stretch",
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            "REPLACE IMAGE",
                            key=f"replace_image_{i}",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.video_data["images"][i]["image_path"] = (
                                st.session_state.new_image_data[i]["image_path"]
                            )
                            st.session_state.video_data["images"][i][
                                "custom_image"
                            ] = None
                            st.session_state.script_df.at[i, "custom_image"] = None
                            st.session_state.new_image_data.pop(i)
                            st.rerun()

                    with col2:
                        if st.button(
                            "CANCEL",
                            key=f"cancel_image_{i}",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.new_image_data.pop(i)
                            st.rerun()
                else:
                    custom_reference_imgs = []
                    custom_regenerate_text = ""
                    with st.expander("REGENRATE IMAGE WITH CUSTOM INSTRUCTIONS"):
                        if st.session_state.image_refinement_mode:
                            custom_reference_imgs = st.file_uploader(
                                "Upload Custom Reference Images (2 MAX)",
                                type="png",
                                accept_multiple_files=True,
                                key=f"custom_reference_imgs_{i}",
                            )
                            if custom_reference_imgs:
                                if len(custom_reference_imgs) > 2:
                                    st.warning(
                                        "⚠️ You can upload up to 2 images only. "
                                        f"You uploaded {len(custom_reference_imgs)}."
                                    )
                                    custom_reference_imgs = custom_reference_imgs[
                                        :2
                                    ]  # keep only first N

                        custom_regenerate_text = st.text_area(
                            "Custom Image Generation Instructions",
                            placeholder="Enter image generation instructions here",
                            disabled=st.session_state.generating,
                            key=f"regenerate_text_{i}",
                            max_chars=300,
                        )

                        if st.button(
                            "REGENERATE IMAGE",
                            key=f"regenerate_{i}",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            with st.spinner(f"Regenerating Scene {i + 1}..."):
                                try:
                                    item = {
                                        "scene": entry["scene"],
                                        "video_scene": entry["video_scene"],
                                        "script": entry["script"],
                                        "custom_image": None,
                                        "reference_images": custom_reference_imgs,
                                        "reference_text": custom_regenerate_text,
                                        "old_image": (
                                            entry.get("custom_image")
                                            if entry.get("custom_image")
                                            else entry.get("image_path")
                                        ),
                                    }
                                    new_img_gen = _generate_storyboard_images(
                                        [item], global_dimension, model_type
                                    )
                                    new_img = next(new_img_gen)

                                    st.session_state.new_image_data[i] = {
                                        "image_path": new_img["image_path"]
                                    }
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"❌ Failed to regenerate image: {e}")


def normalize_veo3_video(input_path):
    normalized_path = Path(tempfile.gettempdir()) / f"normalized_{uuid.uuid4()}.mp4"
    (
        ffmpeg.input(str(input_path))
        .output(
            str(normalized_path),
            vcodec="libx264",
            acodec="aac",
            crf=23,
            pix_fmt="yuv420p",
            movflags="+faststart",
        )
        .overwrite_output()
        .run()
    )
    return normalized_path


def watermark_addition(final_output):
    try:
        # Normalize Veo3 video before watermarking
        normalized_video = normalize_veo3_video(final_output)

        final_watermarked = (
            Path(tempfile.gettempdir()) / f"final_watermarked_{uuid.uuid4()}.mp4"
        )

        # Input video with audio
        video_input = ffmpeg.input(str(normalized_video))
        watermark_input = ffmpeg.input(WATERMARK)

        # Step 1: Overlay watermark on video stream only
        watermarked_video = (
            ffmpeg.filter(
                [video_input.video, watermark_input],
                "overlay",
                x="(main_w-overlay_w)/2",
                y="(main_h-overlay_h)/2",
            )
            .filter("format", "rgba")
            .filter("colorchannelmixer", aa=0.9)
        )

        # # only add audio if present
        # streams = [watermarked_video]
        # try:
        #     streams.append(video_input.audio)
        # except Exception:
        #     pass  # no audio track

        # Probe input to see if it has audio
        probe = ffmpeg.probe(str(normalized_video))
        has_audio = any(stream["codec_type"] == "audio" for stream in probe["streams"])

        if has_audio:
            ffmpeg.output(
                watermarked_video,
                video_input.audio,
                str(final_watermarked),
                vcodec="libx264",  # re-encode video with watermark
                acodec="aac",  # re-encode audio
                crf=17,
                preset="slow",
                audio_bitrate="192k",
                pix_fmt="yuv420p",
                movflags="+faststart",
            ).overwrite_output().run()
        else:
            ffmpeg.output(
                watermarked_video,
                str(final_watermarked),
                vcodec="libx264",  # re-encode video with watermark
                acodec="aac",  # re-encode audio
                crf=17,
                preset="slow",
                audio_bitrate="192k",
                pix_fmt="yuv420p",
                movflags="+faststart",
            ).overwrite_output().run()

        final_output = final_watermarked  # Update to the new watermarked version

    except ffmpeg.Error as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        st.warning("⚠️ Failed to add watermark to final output.")
        logger.error("Watermark error:\n" + err_msg)

    return final_output


def logo_addition(video_path, logo_path, position="top-right"):
    """
    Add a logo to the given video.

    Args:
        video_path (str or Path): Path to the input video.
        logo_path (str or Path): Path to the logo image.
        position (str): Either "top-right" or "top-left".

    Returns:
        Path: Path to the video with logo overlay.
    """
    try:
        video_path = Path(video_path)
        logo_path = Path(logo_path)

        final_logo_video = (
            Path(tempfile.gettempdir()) / f"final_logo_{uuid.uuid4()}.mp4"
        )

        normalized_video = normalize_veo3_video(video_path)

        # Input video and logo
        video_input = ffmpeg.input(str(normalized_video))
        logo_input = ffmpeg.input(str(logo_path)).filter(
            "scale", -1, 30
        )  # -1 means auto width

        # Position logic
        if position == "top-right":
            x_pos = "(main_w-overlay_w-25)"  # 25px padding from right
            y_pos = "10"  # 10px padding from top
        elif position == "top-left":
            x_pos = "10"  # 10px padding from left
            y_pos = "10"  # 10px padding from top
        else:
            raise ValueError("Invalid position. Use 'top-right' or 'top-left'.")

        # Overlay logo on video
        video_with_logo = ffmpeg.overlay(
            video_input.video, logo_input, x=x_pos, y=y_pos
        )

        # # only add audio if present
        # streams = [video_with_logo]
        # try:
        #     streams.append(video_input.audio)
        # except Exception:
        #     pass  # no audio track

        # Probe input to see if it has audio
        probe = ffmpeg.probe(str(video_path))
        has_audio = any(stream["codec_type"] == "audio" for stream in probe["streams"])

        if has_audio:
            # Merge with original audio
            ffmpeg.output(
                video_with_logo,
                video_input.audio,
                str(final_logo_video),
                vcodec="libx264",
                acodec="aac",
                crf=18,
                preset="slow",
                pix_fmt="yuv420p",
                movflags="+faststart",
            ).overwrite_output().run()
        else:
            ffmpeg.output(
                video_with_logo,
                str(final_logo_video),
                vcodec="libx264",
                acodec="aac",
                crf=18,
                preset="slow",
                pix_fmt="yuv420p",
                movflags="+faststart",
            ).overwrite_output().run()

        return final_logo_video

    except ffmpeg.Error as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error("Logo addition error:\n" + err_msg)
        raise RuntimeError("Failed to add logo to video") from e


def _generate_single_video(scene, ref_images, model_type, dimension):

    temp_file = (
        Path(tempfile.gettempdir())
        / f"scene_{scene['scene_number']}_{uuid.uuid4()}.mp4"
    )

    prompt = f"Script: {scene['script']}\nImage Scene: {scene['scene_text']}\nVideo Scene: {scene['video_scene_text']}"

    if model_type == "openai":
        sora_video_generation_pipeline(
            image_path=scene.get("image_path"),
            prompt=prompt,
            dimension=SORA_DIMENSIONS.get(dimension),
            duration=int(scene.get("duration")),
            output_path=temp_file,
        )

    elif model_type == "gemini":
        with open(str(scene["image_path"]), "rb") as f:
            img_bytes = f.read()
        image = types.Image(mime_type="image/png", image_bytes=img_bytes)

        config = types.GenerateVideosConfig(
            number_of_videos=1,
            # resolution="1080p",
            # enhance_prompt=True,
            # generate_audio=False,
            # person_generation="allow_adult",
            aspect_ratio=ASPECT_RATIOS[dimension],
        )
        operation = GEMINI_CLIENT.models.generate_videos(
            model="veo-3.0-generate-preview", prompt=prompt, image=image, config=config
        )
        # operation = GEMINI_CLIENT.models.generate_videos(
        #     model="veo-2.0-generate-001", prompt=prompt, image=image, config=config
        # )
        while not operation.done:
            operation = GEMINI_CLIENT.operations.get(operation)
            time.sleep(5)

        if not operation.response or not operation.response.generated_videos:
            logger.error("No video was generated.")
            logger.error("Raw operation result:", operation)
        video = operation.response.generated_videos[0]
        GEMINI_CLIENT.files.download(file=video.video)
        video.video.save(temp_file)

    # elif model_type == "fastwan":
    #     width, height = FASTWAN_DIMENSIONS[dimension]
    #     cropped_image_path = crop_image_to_dimension(
    #         scene["image_path"], f"{width}x{height}"
    #     )
    #     with open(cropped_image_path, "rb") as f:
    #         img_bytes = f.read()
    #     image_b64 = base64.b64encode(img_bytes).decode("utf-8")
    #     fastwan_video_generation(
    #         prompt, height, width, scene["duration"], image_b64, temp_file
    #     )

    else:
        return ""

    if temp_file.exists():
        if st.session_state.use_logo:
            if st.session_state.custom_logo:
                uploaded_image_path = save_uploaded_file(st.session_state.custom_logo)
                temp_file = logo_addition(
                    temp_file, uploaded_image_path, st.session_state.logo_location
                )

        if Path(WATERMARK).exists() and st.session_state.watermark:
            temp_file = watermark_addition(temp_file)

        return str(temp_file)
    return ""


def generate_video(video_data, global_duration, global_dimension, model_type):
    """
    Streamlined video generation with live status updates, per-scene progress,
    and a final merge button for audio/video.
    """

    if "scene_videos" not in st.session_state:
        st.session_state.scene_videos = dict()
    if "scene_video_data" not in st.session_state:
        st.session_state.scene_video_data = []
    # Parse scenes timestamps
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    video_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(
        0, text="📑 Preparing scenes metadata..."
    )
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
    # total_duration = sum(s["duration"] for s in scenes)
    # total_gap = max(global_duration - total_duration, 0)
    # GAP = total_gap / (len(scenes) - 1) if len(scenes) > 1 else 0
    # audio = AudioSegment.from_file(video_data["audio_path"])
    # status_box.write("✅ Audio Loaded...")

    st.session_state.scene_video_data = scenes
    scene_count = len(scenes)

    # Generate each scene with per-scene progress bar
    with video_placeholder.expander("🎬 Preview Generated Scene Videos"):
        for i, scene in enumerate(scenes):
            status_box.write(
                f"🎬 Generating Scene {scene['scene_number']} of {scene_count}..."
            )
            previous_image = None if i == 0 else scenes[i - 1]["image_path"]
            temp_file = _generate_single_video(
                scene, ref_images, model_type, global_dimension
            )
            st.session_state.scene_videos[scene["scene_number"]] = {
                "script": scene["script"],
                "scene_text": scene["scene_text"],
                "video_scene_text": scene["video_scene_text"],
                "video_path": str(temp_file),
            }
            # Display generated content
            if st.session_state.scene_videos[scene["scene_number"]]:
                with st.expander(f"Scene {scene['scene_number']}"):
                    st.text_area("Script", scene["script"], disabled=True)
                    st.text_area("Scene Text", scene["scene_text"], disabled=True)
                    st.text_area(
                        "Video Scene Text", scene["video_scene_text"], disabled=True
                    )
                    video_path = st.session_state.scene_videos[scene["scene_number"]][
                        "video_path"
                    ]
                    if video_path:
                        st.video(video_path)
                    else:
                        st.error("❌ Video generation failed for this scene.")

            progress_bar.progress((scene["scene_number"]) / scene_count)
            status_box.write(f"✅ Scene {scene['scene_number']} Generated")

        progress_bar.progress(100, text="✅ Scene Videos Generated Successfully.")
        status_box.update(label="✅ Complete", state="complete")

    progress_placeholder.empty()
    status_placeholder.empty()
    video_placeholder.empty()


def video_gallery(global_dimension, model_type):
    if "new_video_data" not in st.session_state:
        st.session_state.new_video_data = {}
    with st.expander("Preview Generated Scene Videos", expanded=True):
        for scene, value in st.session_state.scene_videos.items():
            with st.expander(f"Scene {scene}"):
                st.text_area(
                    "Script",
                    value["script"],
                    disabled=True,
                    key=f"generated_video_script_{scene}",
                )
                st.text_area(
                    "Scene Text",
                    value["scene_text"],
                    disabled=True,
                    key=f"generated_video_{scene}",
                )
                st.text_area(
                    "Video Scene Text",
                    value["video_scene_text"],
                    disabled=True,
                    key=f"generated_video_scene_{scene}",
                )
                video_path = value["video_path"]
                if video_path:
                    st.video(video_path)
                else:
                    st.error("❌ Video generation failed for this scene.")

                if scene in st.session_state.new_video_data:
                    st.divider()
                    st.write("New Regenerated Scene Video:")
                    new_video_path = st.session_state.new_video_data[scene][
                        "video_path"
                    ]
                    if new_video_path:
                        st.video(new_video_path)
                    else:
                        st.error("❌ Video generation failed for this scene.")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            f"REPLACE VIDEO",
                            key=f"replace_video_{scene}",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.scene_videos[scene]["video_path"] = (
                                st.session_state.new_video_data[scene]["video_path"]
                            )
                            st.session_state.new_video_data.pop(scene)
                            st.rerun()

                    with col2:
                        if st.button(
                            f"CANCEL",
                            key=f"cancel_video_{scene}",
                            width="stretch",
                            disabled=st.session_state.generating,
                        ):
                            st.session_state.new_video_data.pop(scene)
                            st.rerun()
                else:
                    if st.button(
                        "REGENERATE SCENE VIDEO",
                        key=f"regen_video_{scene}",
                        width="stretch",
                        disabled=st.session_state.generating,
                    ):

                        with st.spinner("Regenerating scene video..."):
                            try:
                                scene_data = st.session_state.scene_video_data[
                                    scene - 1
                                ]
                                new_file = _generate_single_video(
                                    scene_data, "", model_type, global_dimension
                                )

                                st.session_state.new_video_data[scene] = {
                                    "video_path": new_file,
                                    "script": scene_data["script"],
                                    "scene_text": scene_data["scene_text"],
                                    "video_scene_text": scene_data["video_scene_text"],
                                }
                                st.rerun()
                            except Exception as e:
                                st.error(
                                    f"❌ Failed to regenerate scene {scene} video: {e}"
                                )


def sync_so_lipsync_pipeline(
    audio_url: str, video_url: str, output_dir: str | None = None
):
    if not (audio_url or video_url):
        logger.error("Please Provide Valid Video and Audio URL")
        return None

    logger.info("Starting lip sync generation job...")
    try:
        response = SYNC_SO_CLIENT.generations.create(
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

        generation = SYNC_SO_CLIENT.generations.get(job_id)
        status = generation.status

        while status not in ["COMPLETED", "FAILED", "REJECTED"]:
            logger.info(f"Polling status for generation {job_id}")
            time.sleep(10)
            generation = SYNC_SO_CLIENT.generations.get(job_id)
            status = generation.status

        if status == "COMPLETED":
            output_url = generation.output_url
            logger.info(
                f"Generation {job_id} completed successfully, output url: {output_url}"
            )

            if output_url:
                # Create a temporary file with .mp4 extension
                if output_dir and os.path.exists(output_dir):
                    temp_fd, temp_path = tempfile.mkstemp(
                        suffix=".mp4", dir=output_dir, prefix="lipsync_"
                    )
                else:
                    temp_fd, temp_path = tempfile.mkstemp(
                        suffix=".mp4", prefix="lipsync_"
                    )

                logger.info(f"Created temporary file: {temp_path}")

                try:
                    # Download the video file
                    logger.info(f"Downloading video from {output_url}")
                    response = requests.get(output_url, stream=True, timeout=300)
                    response.raise_for_status()

                    # Write to the temporary file
                    with os.fdopen(temp_fd, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    logger.info(f"Video successfully downloaded to {temp_path}")
                    return temp_path

                except Exception as download_error:
                    # Clean up the temp file if download fails
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    logger.error(f"Error downloading video: {download_error}")
                    return None
            else:
                logger.error(f"Generation {job_id} - No Output URL in Response")
                return None
        else:
            logger.error(f"Generation {job_id} failed with status: {status}")
            return None

    except ApiError as e:
        logger.error(
            f"Create generation request failed with status code {e.status_code} and error {e.body}"
        )
        return None
    except Exception as e:
        logger.error(f"Error Occurred During LipSync Generation: {str(e)}")
        return None


def split_media_into_chunks(file_path: str, max_duration: float = 299) -> list:
    """
    Split media file into chunks of max_duration seconds
    Returns list of tuples: (start_time, end_time, chunk_path)
    """
    total_duration = get_file_duration(file_path)

    if total_duration <= max_duration:
        # No need to split
        return [(0, total_duration, file_path)]

    chunks = []
    num_chunks = int((total_duration) / (max_duration)) + 1

    for i in range(num_chunks):
        start_time = i * (max_duration)
        end_time = min(start_time + max_duration, total_duration)

        # Create chunk file path
        file_stem = Path(file_path).stem
        file_ext = Path(file_path).suffix
        chunk_path = Path(tempfile.gettempdir()) / f"{file_stem}_chunk_{i}{file_ext}"

        # Trim the file
        progress = st.empty()
        if trim_media(file_path, start_time, end_time, str(chunk_path), progress):
            chunks.append((start_time, end_time, str(chunk_path)))

    return chunks


def merge_videos(video_chunks: list, output_path: str) -> bool:
    """
    Merge multiple video chunks using ffmpeg concat demuxer
    video_chunks: list of video file paths
    """
    try:
        # Create concat file
        concat_file = Path(tempfile.gettempdir()) / "concat_list.txt"
        with open(concat_file, "w") as f:
            for chunk in video_chunks:
                f.write(f"file '{chunk}'\n")

        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",  # Copy without re-encoding
            "-y",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            concat_file.unlink()  # Clean up concat file
            return True
        else:
            st.error(f"Merge error: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"Error merging videos: {str(e)}")
        return False


def lipsync_generation_pipeline(video_path: str, audio_path: str):
    """
    Main pipeline with chunk splitting for files > 299 seconds
    """
    video_duration = get_file_duration(video_path)
    audio_duration = get_file_duration(audio_path)

    st.info(
        f"Video duration: {format_time(video_duration)} | Audio duration: {format_time(audio_duration)}"
    )
    # Check if splitting is needed
    needs_split = video_duration > 299 or audio_duration > 299

    if needs_split:

        # Split media into chunks
        video_chunks = split_media_into_chunks(video_path, 299)
        audio_chunks = split_media_into_chunks(audio_path, 299)
        st.info(
            f"Split into {len(video_chunks)} video chunks and {len(audio_chunks)} audio chunks"
        )

        synced_chunks = []

        # Process each chunk pair
        for idx, ((v_start, v_end, v_chunk), (a_start, a_end, a_chunk)) in enumerate(
            zip(video_chunks, audio_chunks)
        ):

            # Upload chunks to static server
            video_url = upload_file_to_static_server(v_chunk)
            audio_url = upload_file_to_static_server(a_chunk)

            if not video_url or not audio_url:
                continue

            # Process chunk through lipsync pipeline
            synced_chunk_path = sync_so_lipsync_pipeline(
                audio_url=audio_url,
                video_url=video_url,
                output_dir=str(Path(tempfile.gettempdir())),
            )

            if synced_chunk_path:
                synced_chunks.append(synced_chunk_path)
            else:
                return None

        # Merge all synced chunks
        if synced_chunks:
            merged_output_path = Path(tempfile.gettempdir()) / "final_merged_video.mp4"

            if merge_videos(synced_chunks, str(merged_output_path)):

                # Clean up chunk files
                for chunk in (
                    synced_chunks
                    + [c[2] for c in video_chunks]
                    + [c[2] for c in audio_chunks]
                ):
                    try:
                        Path(chunk).unlink()
                    except:
                        pass

                return merged_output_path
            else:
                return None
    else:
        video_url = upload_file_to_static_server(video_path)
        if not video_url:
            st.error("Failed to upload video to static server")
            return None

        audio_url = upload_file_to_static_server(audio_path)
        if not audio_url:
            st.error("Failed to upload audio to static server")
            return None

        lipsynced_video_path = sync_so_lipsync_pipeline(
            audio_url=audio_url,
            video_url=video_url,
            output_dir=str(Path(tempfile.gettempdir())),
        )

        return lipsynced_video_path


def final_generation(video_data, use_custom_audio, final_quality):
    with st.status("🔗 Merging scenes into one final video...") as status:
        try:
            merged_tmp = Path(tempfile.gettempdir()) / f"merged_{uuid.uuid4()}.mp4"
            # Load inputs and prepare input streams
            input_streams = []
            has_audio = False

            for scene_info in st.session_state.scene_videos.values():
                video_path = str(scene_info["video_path"])
                inp = ffmpeg.input(video_path)

                # Always add video
                input_streams.append(inp.video)

                # Probe to check if audio exists
                probe = ffmpeg.probe(video_path)
                if any(stream["codec_type"] == "audio" for stream in probe["streams"]):
                    input_streams.append(inp.audio)
                    has_audio = True

            # Concat, conditionally including audio
            concat = ffmpeg.concat(*input_streams, v=1, a=1 if has_audio else 0).node
            v = concat[0]
            a = concat[1] if has_audio else None

            if has_audio:
                out = ffmpeg.output(v, a, str(merged_tmp))
            else:
                out = ffmpeg.output(v, str(merged_tmp))

            out = out.overwrite_output()
            out.run(capture_stdout=True, capture_stderr=True)

            # Final output path
            final_output = Path(tempfile.gettempdir()) / f"final_{uuid.uuid4()}.mp4"
            video_input = ffmpeg.input(str(merged_tmp))
            audio_attached = False

            if use_custom_audio:
                audio_path = video_data.get("audio_path")

                if (
                    st.session_state.lipsync_mode
                    and audio_path
                    and Path(audio_path).exists()
                ):
                    # LIPSYNC MODE: Upload files to static server and use sync.so
                    status.update(
                        label="🎬 Starting lip sync (this may take a few minutes)...",
                        state="running",
                    )

                    lipsynced_video_path = lipsync_generation_pipeline(
                        video_path=str(merged_tmp), audio_path=audio_path
                    )

                    if lipsynced_video_path and Path(lipsynced_video_path).exists():
                        # Apply final quality settings to the lipsynced video
                        ffmpeg.output(
                            ffmpeg.input(lipsynced_video_path),
                            str(final_output),
                            vcodec="libx264",
                            acodec="aac",
                            audio_bitrate="300k",
                            pix_fmt="yuv420p",
                            movflags="+faststart",
                            vf=final_quality,
                        ).overwrite_output().run(
                            capture_stdout=True, capture_stderr=True
                        )

                        # Clean up temp lipsynced file
                        Path(lipsynced_video_path).unlink(missing_ok=True)
                        audio_attached = True
                        status.update(
                            label="✅ Lip sync completed successfully!",
                            state="complete",
                        )
                    else:
                        st.error(
                            "Lip sync processing failed. Using original video with audio."
                        )
                        # Fallback to regular audio attachment
                        audio_input = ffmpeg.input(audio_path)
                        ffmpeg.output(
                            video_input.video,
                            audio_input,
                            str(final_output),
                            vcodec="libx264",
                            acodec="aac",
                            audio_bitrate="300k",
                            pix_fmt="yuv420p",
                            movflags="+faststart",
                            vf=final_quality,
                            **{"map": "0:v:0", "map": "1:a:0"},
                        ).overwrite_output().run(
                            capture_stdout=True, capture_stderr=True
                        )
                        audio_attached = True

                elif audio_path and Path(audio_path).exists():
                    # REGULAR MODE: Just attach audio without lip sync
                    status.update(label="🎵 Attaching audio...", state="running")
                    audio_input = ffmpeg.input(audio_path)
                    ffmpeg.output(
                        video_input.video,
                        audio_input,
                        str(final_output),
                        vcodec="libx264",
                        acodec="aac",
                        audio_bitrate="300k",
                        pix_fmt="yuv420p",
                        movflags="+faststart",
                        vf=final_quality,
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
                    vcodec="libx264",
                    acodec="copy",
                    pix_fmt="yuv420p",
                    movflags="+faststart",
                    vf=final_quality,
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)

            st.session_state.final_output_path = str(final_output)
            st.session_state.video_generated = True

            # Clean up merged temp file
            Path(merged_tmp).unlink(missing_ok=True)
            status.update(
                label="✅ Final video generated successfully!", state="complete"
            )

        except ffmpeg.Error as e:
            err_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error("⚠️ FFmpeg error:", err_msg)
            st.error("FFmpeg failed while merging videos.")
            status.update(label="❌ Video generation failed", state="error")
