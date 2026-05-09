"""Watermark Remover tab."""

from __future__ import annotations

import os
import tempfile

import cv2
import streamlit as st

from services.media.watermark import (
    extract_frame,
    remove_watermark_ffmpeg,
    remove_watermark_opencv,
)


def render() -> None:
    st.title("🎬 :red[**Video Watermark Remover**] ✂️")

    if "processed_video_path" not in st.session_state:
        st.session_state.processed_video_path = None
    if "original_video_path" not in st.session_state:
        st.session_state.original_video_path = None
    if "coordinates" not in st.session_state:
        st.session_state.coordinates = None
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False

    uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov", "mkv"])

    if not uploaded_file:
        st.info("👆 Upload a video to get started")
        st.markdown(
            """
        ### How to use:
        1. **Upload your video** with the watermark
        2. **Choose removal method** - OpenCV for better quality or FFmpeg for speed
        3. **Select watermark position** - Choose a corner preset or enter custom coordinates
        4. **Adjust size** - Set the width and height of the watermark area
        5. **Preview** - Check the red rectangle covers your watermark
        6. **Remove** - Click the button and download your cleaned video!

        ### Method Comparison:
        - **OpenCV Inpainting**: More natural results, reconstructs textures intelligently
          - *Telea*: Best for textured/patterned backgrounds
          - *Navier-Stokes*: Best for smooth gradients and edges
        - **FFmpeg Delogo**: Faster processing but creates more blur

        ### Tips:
        - Use the **corner presets** for watermarks in standard positions
        - Try **OpenCV Inpainting** first for best quality
        - Adjust the **margin** if watermark isn't exactly at the edge
        - Make the selection **slightly larger** than the watermark for best results
        - Test with a short video first to verify settings
        """
        )
        return

    if st.session_state.original_video_path is None or not os.path.exists(
        st.session_state.original_video_path
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state.original_video_path = tmp_file.name

    temp_input_path = st.session_state.original_video_path

    st.subheader("📍 Step 1: Select Watermark Area")
    frame = extract_frame(temp_input_path)

    if frame is not None:
        st.markdown("### Watermark Position")
        height, width = frame.shape[:2]
        st.write(f"**Video Size:** {width} x {height} px")
        st.markdown("---")
        st.markdown("**⚙️ Settings**")

        removal_method = st.radio(
            "Removal Method:",
            [
                "OpenCV Inpainting (Better Quality, Slower)",
                "FFmpeg Delogo (Faster, More Blur)",
            ],
            help="OpenCV provides more natural results but takes longer",
            key="method_select",
        )

        if "OpenCV" in removal_method:
            inpaint_algo = st.selectbox(
                "Inpainting Algorithm:",
                ["Telea (Better for textures)", "Navier-Stokes (Better for edges)"],
                help="Telea works best for textured backgrounds, Navier-Stokes for smooth gradients",
                key="inpaint_algo",
            )

        corner = st.selectbox(
            "Choose corner:",
            ["Custom", "Bottom Right", "Bottom Left", "Top Right", "Top Left"],
            key="corner_select",
        )

        wm_width = st.number_input("Watermark Width (px)", 50, width, 50, key="wm_width")
        wm_height = st.number_input("Watermark Height (px)", 20, height, 25, key="wm_height")
        margin = st.number_input("Margin from edge (px)", 0, 100, 30, key="margin")

        if corner == "Bottom Right":
            x = width - wm_width - margin
            y = height - wm_height - margin
        elif corner == "Bottom Left":
            x = margin
            y = height - wm_height - margin
        elif corner == "Top Right":
            x = width - wm_width - margin
            y = margin
        elif corner == "Top Left":
            x = margin
            y = margin
        else:
            st.markdown("**Manual Entry:**")
            x = st.number_input("X (left position)", 0, width, 0, key="x_pos")
            y = st.number_input("Y (top position)", 0, height, 0, key="y_pos")

        st.session_state.coordinates = (x, y, wm_width, wm_height)

        st.markdown("---")
        st.markdown(
            f"**Selected Region: | X: {int(x)} | Y: {int(y)} | "
            f"Width: {wm_width} | Height: {wm_height} |**"
        )

        preview_frame = frame.copy()
        cv2.rectangle(
            preview_frame,
            (int(x), int(y)),
            (int(x + wm_width), int(y + wm_height)),
            (255, 0, 0), 3,
        )
        st.image(preview_frame, caption="Preview with Selection", width="stretch")

    st.markdown("---")
    if st.button("Remove Watermark", type="primary", width="stretch"):
        with st.spinner("Processing video... This may take a few minutes."):
            try:
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                if "OpenCV" in removal_method:
                    method = "telea" if "Telea" in inpaint_algo else "ns"
                    remove_watermark_opencv(
                        temp_input_path, output_path,
                        int(x), int(y), int(wm_width), int(wm_height),
                        method,
                    )
                else:
                    remove_watermark_ffmpeg(
                        temp_input_path, output_path,
                        int(x), int(y), int(wm_width), int(wm_height),
                    )

                st.session_state.processed_video_path = output_path
                st.session_state.processing_done = True
                st.success("✅ Watermark removed successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                st.info(
                    "Make sure FFmpeg and OpenCV are installed: "
                    "`pip install ffmpeg-python opencv-python numpy`"
                )

    if st.session_state.processing_done and st.session_state.processed_video_path:
        st.markdown("---")
        st.subheader("✨ Step 2: Preview & Download")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🎥 Original Video**")
            if os.path.exists(st.session_state.original_video_path):
                st.video(st.session_state.original_video_path)
        with col2:
            st.markdown("**✅ Processed Video (Watermark Removed)**")
            if os.path.exists(st.session_state.processed_video_path):
                st.video(st.session_state.processed_video_path)

        st.markdown("---")

        if os.path.exists(st.session_state.processed_video_path):
            with open(st.session_state.processed_video_path, "rb") as f:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=f,
                    file_name="video_no_watermark.mp4",
                    mime="video/mp4",
                    width="stretch",
                )

        if st.button("🔄 Process Another Video", width="stretch"):
            if st.session_state.processed_video_path and os.path.exists(
                st.session_state.processed_video_path
            ):
                os.unlink(st.session_state.processed_video_path)
            if st.session_state.original_video_path and os.path.exists(
                st.session_state.original_video_path
            ):
                os.unlink(st.session_state.original_video_path)

            st.session_state.processed_video_path = None
            st.session_state.original_video_path = None
            st.session_state.coordinates = None
            st.session_state.processing_done = False
            st.rerun()
