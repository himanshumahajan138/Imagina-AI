"""YouTube Downloader tab."""

from __future__ import annotations

import streamlit as st

from services.media.trimmer import display_media_player
from services.media.youtube import is_valid_youtube_url, youtube_downloader_pipeline


def _kick_download(youtube_url: str, kind: str) -> None:
    if not youtube_url:
        st.error("❌ Please enter a YouTube URL")
        return
    if not is_valid_youtube_url(youtube_url):
        st.error("❌ Please enter a valid YouTube URL")
        return
    spinner_label = {"audio": "audio", "video": "video", "mix": "mix", "both": "both files"}[kind]
    with st.spinner(f"Downloading {spinner_label}..."):
        st.session_state.download_result = youtube_downloader_pipeline(youtube_url, kind)
        st.session_state.result_type = kind


def render() -> None:
    st.title("🎬 :red[**YouTube Downloader**] 📽️")

    if "download_result" not in st.session_state:
        st.session_state.download_result = None
    if "result_type" not in st.session_state:
        st.session_state.result_type = None

    youtube_url = st.text_input(
        "📌 Enter YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste the full YouTube URL here",
    )

    st.subheader("⬇️ Download Options")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🎵 Audio Only", use_container_width=True, key="btn_audio"):
            _kick_download(youtube_url, "audio")
    with col2:
        if st.button("🎬 Video Only", use_container_width=True, key="btn_video"):
            _kick_download(youtube_url, "video")
    with col3:
        if st.button("🎬🎵 Mix", use_container_width=True, key="btn_mix"):
            _kick_download(youtube_url, "mix")
    with col4:
        if st.button("📦 Both", use_container_width=True, key="btn_both"):
            _kick_download(youtube_url, "both")

    st.divider()

    if not st.session_state.download_result:
        return

    result = st.session_state.download_result
    result_type = st.session_state.result_type

    if not result["success"]:
        st.error(result["error"])
        return

    if result_type == "audio":
        st.success("✅ Audio extracted successfully!")
        audio_path = result["data"].get("audio_path")
        if audio_path:
            try:
                display_media_player(audio_path, audio_path)
                with open(audio_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Audio",
                        data=f,
                        file_name=audio_path.split("/")[-1],
                        mime="audio/mpeg",
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"Could not load preview: {str(e)}")

    elif result_type == "video":
        st.success("✅ Video extracted successfully!")
        video_path = result["data"].get("video_path")
        if video_path:
            try:
                display_media_player(video_path, video_path)
                with open(video_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Video",
                        data=f,
                        file_name=video_path.split("/")[-1],
                        mime="video/mp4",
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"Could not load preview: {str(e)}")

    elif result_type == "mix":
        st.success("✅ Mix extracted successfully!")
        hybrid_path = result["data"].get("hybrid_path")
        if hybrid_path:
            try:
                display_media_player(hybrid_path, hybrid_path)
                with open(hybrid_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Mix",
                        data=f,
                        file_name=hybrid_path.split("/")[-1],
                        mime="video/mp4",
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"Could not load preview: {str(e)}")

    elif result_type == "both":
        st.success("✅ Both files extracted successfully!")
        audio_path = result["data"].get("audio_path")
        video_path = result["data"].get("video_path")

        st.subheader("🎵 Audio File")
        if audio_path:
            try:
                display_media_player(audio_path, audio_path)
                with open(audio_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Audio",
                        data=f,
                        file_name=audio_path.split("/")[-1],
                        mime="audio/mpeg",
                        use_container_width=True,
                        key="download_audio_both",
                    )
            except Exception as e:
                st.warning(f"Could not load audio preview: {str(e)}")

        st.subheader("🎬 Video File")
        if video_path:
            try:
                display_media_player(video_path, video_path)
                with open(video_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Video",
                        data=f,
                        file_name=video_path.split("/")[-1],
                        mime="video/mp4",
                        use_container_width=True,
                        key="download_video_both",
                    )
            except Exception as e:
                st.warning(f"Could not load video preview: {str(e)}")
