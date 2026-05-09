"""Watermark + logo + frame helpers (OpenCV + ffmpeg)."""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
import streamlit as st
from PIL import Image

from core.logger import logger


WATERMARK = "images/watermark.png"


def extract_frame(video_path, frame_number=0):
    """Extract a frame from video for preview."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def remove_watermark_opencv(input_path, output_path, x, y, width, height, method="telea"):
    """Remove watermark using OpenCV inpainting; preserves audio."""
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".aac").name
    subprocess.run(
        ["ffmpeg", "-i", input_path, "-vn", "-acodec", "copy", audio_path, "-y"],
        capture_output=True,
    )

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (frame_width, frame_height))

    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    mask[y : y + height, x : x + width] = 255

    inpaint_method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inpainted = cv2.inpaint(frame, mask, 7, inpaint_method)
        out.write(inpainted)

    cap.release()
    out.release()

    subprocess.run(
        [
            "ffmpeg", "-i", temp_video, "-i", audio_path,
            "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
            "-shortest", output_path, "-y",
        ],
        check=True,
        capture_output=True,
    )

    if os.path.exists(temp_video):
        os.unlink(temp_video)
    if os.path.exists(audio_path):
        os.unlink(audio_path)


def remove_watermark_ffmpeg(input_path, output_path, x, y, width, height):
    """Remove watermark using FFmpeg delogo (faster, blurrier)."""
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", f"delogo=x={x}:y={y}:w={width}:h={height}",
        "-c:a", "copy", output_path, "-y",
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def crop_image_to_dimension(image_path: str, target_dimension: str) -> str:
    """Center-crop an image to match a target dimension string like '1280x720'."""
    target_width, target_height = map(int, target_dimension.split("x"))
    target_aspect = target_width / target_height

    logger.info(f"Cropping image to {target_dimension}")

    img = Image.open(image_path)
    img_width, img_height = img.size
    img_aspect = img_width / img_height

    if img_width == target_width and img_height == target_height:
        return image_path

    if img_aspect > target_aspect:
        new_width = int(img_height * target_aspect)
        new_height = img_height
        left = (img_width - new_width) // 2
        top, right, bottom = 0, left + new_width, img_height
    else:
        new_width = img_width
        new_height = int(img_width / target_aspect)
        left, top = 0, (img_height - new_height) // 2
        right, bottom = img_width, top + new_height

    cropped_img = img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((target_width, target_height), Image.LANCZOS)

    temp_fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="cropped_")
    os.close(temp_fd)
    resized_img.save(temp_path)
    logger.info(f"Image cropped and saved: {temp_path}")
    return temp_path


def normalize_veo3_video(input_path):
    """Re-encode VEO output for predictable downstream ffmpeg processing."""
    normalized_path = Path(tempfile.gettempdir()) / f"normalized_{uuid.uuid4()}.mp4"
    (
        ffmpeg.input(str(input_path))
        .output(
            str(normalized_path),
            vcodec="libx264", acodec="aac", crf=23,
            pix_fmt="yuv420p", movflags="+faststart",
        )
        .overwrite_output()
        .run()
    )
    return normalized_path


def watermark_addition(final_output):
    """Overlay the project's centered watermark image on the given video."""
    try:
        normalized_video = normalize_veo3_video(final_output)
        final_watermarked = Path(tempfile.gettempdir()) / f"final_watermarked_{uuid.uuid4()}.mp4"

        video_input = ffmpeg.input(str(normalized_video))
        watermark_input = ffmpeg.input(WATERMARK)

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

        probe = ffmpeg.probe(str(normalized_video))
        has_audio = any(s["codec_type"] == "audio" for s in probe["streams"])

        if has_audio:
            ffmpeg.output(
                watermarked_video, video_input.audio, str(final_watermarked),
                vcodec="libx264", acodec="aac", crf=17, preset="slow",
                audio_bitrate="192k", pix_fmt="yuv420p", movflags="+faststart",
            ).overwrite_output().run()
        else:
            ffmpeg.output(
                watermarked_video, str(final_watermarked),
                vcodec="libx264", acodec="aac", crf=17, preset="slow",
                audio_bitrate="192k", pix_fmt="yuv420p", movflags="+faststart",
            ).overwrite_output().run()

        return final_watermarked
    except ffmpeg.Error as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        st.warning("⚠️ Failed to add watermark to final output.")
        logger.error("Watermark error:\n" + err_msg)
        return final_output


def logo_addition(video_path, logo_path, position="top-right"):
    """Overlay a logo image at top-left or top-right of a video."""
    try:
        video_path = Path(video_path)
        logo_path = Path(logo_path)
        final_logo_video = Path(tempfile.gettempdir()) / f"final_logo_{uuid.uuid4()}.mp4"

        normalized_video = normalize_veo3_video(video_path)

        video_input = ffmpeg.input(str(normalized_video))
        logo_input = ffmpeg.input(str(logo_path)).filter("scale", -1, 30)

        if position == "top-right":
            x_pos, y_pos = "(main_w-overlay_w-25)", "10"
        elif position == "top-left":
            x_pos, y_pos = "10", "10"
        else:
            raise ValueError("Invalid position. Use 'top-right' or 'top-left'.")

        video_with_logo = ffmpeg.overlay(video_input.video, logo_input, x=x_pos, y=y_pos)

        probe = ffmpeg.probe(str(video_path))
        has_audio = any(s["codec_type"] == "audio" for s in probe["streams"])

        if has_audio:
            ffmpeg.output(
                video_with_logo, video_input.audio, str(final_logo_video),
                vcodec="libx264", acodec="aac", crf=18, preset="slow",
                pix_fmt="yuv420p", movflags="+faststart",
            ).overwrite_output().run()
        else:
            ffmpeg.output(
                video_with_logo, str(final_logo_video),
                vcodec="libx264", acodec="aac", crf=18, preset="slow",
                pix_fmt="yuv420p", movflags="+faststart",
            ).overwrite_output().run()

        return final_logo_video
    except ffmpeg.Error as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error("Logo addition error:\n" + err_msg)
        raise RuntimeError("Failed to add logo to video") from e
