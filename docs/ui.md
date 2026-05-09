# UI Layer

Streamlit, organised into a sidebar (global settings) + 5 tabs (one per tool). The cinematic tab orchestrates the main flow; the other four are standalone utilities.

## Files

```
app.py                          Entry point — page_config, sidebar.render, tab fanout, worker health check
ui/
  __init__.py
  sidebar.py                    Global settings: tier picker, voice/dimensions/duration, etc.
  theme.py                      Static theme helpers (logo path, dark-mode toggle)
  tabs/
    cinematic.py                Theme → final video orchestration
    merge.py                    Standalone "merge multiple videos" tool
    watermark.py                "Watermark Remover" tool
    trimmer.py                  "Media Trimmer" tool
    youtube.py                  "YouTube Downloader" tool
  components/
    tier_picker.py              Per-modality model selectbox (sidebar expander)
    storyboard_gallery.py       Per-scene preview + regen widgets used by the cinematic tab
```

## `app.py` — entry point

```python
def main() -> None:
    st.set_page_config(page_title="🎬 Imagina AI Video Generator", layout="wide")
    _bootstrap_session_flags()
    _check_worker()                             # banner if worker daemon is down
    logo_path = theme.resolve_logo_path()
    tab1, ..., tab5 = st.tabs([...])
    sidebar.render(logo_path)                   # always visible
    with tab1: cinematic.render()
    with tab2: merge.render()
    ...

main()
```

`_bootstrap_session_flags` initialises `rerun_needed`, `scene_videos_generated`, `generating`, `action` to `False` if missing. These are the session-state booleans the tabs read in their wizard logic.

`_check_worker` calls `worker.is_alive()` once per session (cached in `st.session_state.worker_alive`) and shows a red error banner with the start-command if down.

## Session-state vocabulary

The cinematic flow uses these keys. (Other tabs have their own; not enumerated here.)

| Key | Type | Set by | Read by |
| --- | --- | --- | --- |
| `preferred_models` | `dict[modality, model_id]` | `tier_picker.render` | `core.registry.session_preferred` (server-side via worker request body) |
| `model_type` | `str` ("openai"/"gemini"/"local") | `sidebar.render` (mirrored from preferred_models["video"]) | legacy callers (LLM prompt seconds, watermark normalize check) |
| `language` | str | sidebar | `pipelines.cinematic._generate_audio` |
| `dimension` | str (label) | sidebar | pipeline + `DIMENSIONS[label]` to resolve |
| `download_quality` | str (label) | sidebar | `final_generation` via `RESOLUTIONS[label]` |
| `duration` | int | sidebar slider | LLM prompt + script validation |
| `watermark`, `use_logo`, `custom_logo`, `logo_location` | toggle / file / str | sidebar | `_generate_single_video` |
| `image_refinement_mode`, `custom_reference_images` | toggle / list | sidebar | `_generate_storyboard_images` |
| `use_custom_audio`, `lipsync_mode`, `selected_speaker`, `selected_speed`, `custom_bgm` | toggle / str / float / file | sidebar | audio pipeline |
| `theme`, `script_file`, `uploaded_content` | text / file / str | sidebar inputs | cinematic tab task handlers |
| `script_df` | DataFrame | LLM gen / SRT parse | cinematic tab data_editor |
| `edited_df`, `last_df_hash`, `image_updated`, `scene_idx`, `custom_image_upload` | various | data_editor + image picker | cinematic tab |
| `generating`, `action` | bool / str | cinematic tab buttons | `_task_handler` to gate execution |
| `editable_images`, `video_data` | list / dict | `_generate_storyboard_images` | storyboard gallery + `generate_video` |
| `scene_videos`, `scene_video_data` | dict / list | `generate_video` | video gallery + `final_generation` |
| `new_image_data`, `new_audio_data`, `new_video_data` | dict | per-scene regen flows | gallery components |
| `video_generated`, `final_output_path` | bool / str | `final_generation` | final preview / download |
| `worker_alive` | bool | `app._check_worker` | banner |

The keys aren't documented anywhere except by reading the code. A future improvement is a typed `SessionState` dataclass mirror.

## `sidebar.py`

Renders the always-visible left rail. Sections:

1. **Logo + header**
2. **Tier picker** (`tier_picker.render()` expander) — see below.
3. **Voice & video settings expander**:
   - Language selectbox
   - Video dimensions selectbox (filtered via `_filtered_dimensions()`)
   - Download resolution selectbox
   - Duration slider (driven by `video_constraints()` from `services.video.service`)
   - Watermark toggle, image refinement toggle, reference-image uploader
   - Use-custom-audio toggle (if on, exposes lipsync toggle, speaker selectbox, speed slider, BGM uploader)
   - Use-logo toggle (if on, exposes logo position + uploader)
4. **Generate Script from Theme expander** — text area + button (writes `theme`, `generate_script_button`).
5. **Upload Existing Script File expander** — file uploader + button + format docs.

The duration slider's min/max/step is dynamic per the active video model:

```python
constraints = video_constraints()           # {scene, min, max, default, step}
st.session_state.duration = st.slider(
    "Duration (seconds)",
    constraints["min"], constraints["max"], constraints["default"],
    step=constraints["step"],
    disabled=st.session_state.generating,
)
```

The dimension dropdown is the **intersection** of `supported_dimensions` across the active image and video models, falling back to all dimensions with a warning if the intersection is empty:

```python
def _filtered_dimensions() -> dict[str, str]:
    allowed = set(common_dimensions(["image", "video"]))
    filtered = {label: val for label, val in DIMENSIONS.items() if val in allowed}
    if not filtered:
        st.warning("⚠️ Selected image and video models share no compatible dimensions; ...")
        return DIMENSIONS
    return filtered
```

`model_type` mirroring (legacy):

```python
video_model = st.session_state.preferred_models.get("video")
if video_model == "veo-3":   st.session_state.model_type = "gemini"
elif video_model == "sora":  st.session_state.model_type = "openai"
else:                        st.session_state.model_type = "local"
```

This exists so legacy callers reading `st.session_state.model_type` (LLM prompt's seconds calc, watermark normalize check) still work. Tier picker remains the single source of truth.

## `components/tier_picker.py`

The "🎚️ Model Selection (advanced)" expander inside the sidebar. One selectbox per modality.

```python
_MODALITIES = [
    ("llm", "📝 Script (LLM)"),
    ("image", "🖼️ Image"),
    ("video", "🎬 Video"),
    ("lipsync", "👄 Lip-sync"),
    ("tts", "🗣️ TTS"),
]
_AUTO_LABEL = "Auto (registry decides)"
```

For each modality:

```python
available_ids = {mid for mid, _ in available_models(modality)}     # env satisfied
all_ids = list(list_models(modality).keys())                       # everything

options = [_AUTO_LABEL] + all_ids
choice = st.selectbox(header, options=options, format_func=_format, ...)

if choice == _AUTO_LABEL:
    st.session_state.preferred_models.pop(modality, None)
else:
    st.session_state.preferred_models[modality] = choice
```

`_format` decorates the option label with badges:

```python
def _format(opt):
    if opt == _AUTO_LABEL: return _AUTO_LABEL
    base = label_for(modality, opt)               # e.g. "z-image-turbo (Local)"
    tags = []
    if opt not in available_ids:                  # env not set
        tags.append("⚠ env not set")
    if is_stub(cfg):                              # `unavailable: true` in yaml
        tags.append("🚧 stub")
    if tags: base += "  ·  " + "  ·  ".join(tags)
    return base
```

So the user sees:

```
Auto (registry decides)
qwen-2.5-7b (Local)
deepseek-v3 (Cloud OSS)  ·  ⚠ env not set
gemini-3.1-flash (API)
gpt-5.4-mini (API)
```

Stubs and env-unsatisfied entries are still selectable — clicking them is the user explicitly opting in. The picker honours preferred over auto-fallback.

## `tabs/cinematic.py`

The main cinematic flow. ~280 lines. Walks through:

1. Two button handlers at the top: `generate_script_button` → set `generating=True; action="generate_script"`, `load_script_button` → similar with `"load_script"`.
2. `_task_handler("generate_script", lambda: ...)` — runs the script gen, then evicts the LLM, resets `generating`, calls `st.rerun()`.
3. `_task_handler("load_script", ...)` — parses the uploaded SRT-style text via `parse_script_scene_content`.
4. `script_df` data editor with column config for speaker/speed/start/end/custom_image.
5. Custom-image-per-scene upload/delete expander.
6. Re-hash detection: `hash_df(edited_df) != last_df_hash` → write back to `script_df`, save new hash, `st.rerun()`.
7. "Generate Scenes" button → `action = "generate_audio_image"` → `_task_handler` calls `generate_audio_images(...)` from the pipeline.
8. After audio/image phase: `storyboard_gallery(...)` (regen / replace per scene).
9. "Generate Final Video" button → `_task_handler` calls `generate_video(...)`.
10. After video phase: `video_gallery(...)` (regen per scene).
11. "Final Merge" button → `_task_handler` calls `final_generation(...)`.
12. Final preview + download button.

The `_task_handler` pattern:

```python
def _task_handler(action_name, task_func):
    if (st.session_state.get("generating")
        and st.session_state.get("action") == action_name):
        task_func()
        st.session_state.generating = False
        st.session_state.action = None
        st.rerun()
```

Pattern: button click sets `generating=True; action=<name>` and reruns. Next rerun, the corresponding `_task_handler` matches the action and runs the heavy work. Done → reset and rerun once more for final UI state.

Why not just call `task_func()` directly in the button-click branch? Streamlit reruns the script on every interaction. Without the two-rerun pattern, the long-running task would run inside the click branch and the UI couldn't show the disabled state of other buttons until the task finished.

## `components/storyboard_gallery.py`

Per-scene preview + interactive widgets used inside the cinematic tab.

`storyboard_gallery(...)` shows each scene's image with:
- "Replace Image" expander (upload custom)
- "Regenerate Image with Custom Instructions" expander (re-prompt)
- Audio regeneration if `use_custom_audio`

It calls back into the pipeline's `_generate_audio` and `_generate_storyboard_images` for individual regens.

`video_gallery(...)` does the equivalent for per-scene videos:
- Per-scene preview
- "Regenerate Scene Video" button — calls `_generate_single_video` on just that scene with the existing seed image

Why these aren't in `tabs/cinematic.py`: they're heavy enough (~330 lines) to warrant their own file, and they're tightly coupled to the pipeline's intermediate state (`editable_images`, `scene_videos`) so they need the same imports.

## `tabs/merge.py`, `watermark.py`, `trimmer.py`, `youtube.py`

Small standalone tools, each in its own tab:

- **merge.py**: upload N videos, ffmpeg concat, download the result.
- **watermark.py**: upload a video, draw a bbox over the watermark, choose OpenCV inpaint or ffmpeg delogo, download cleaned.
- **trimmer.py**: upload audio/video, slider for start/end, ffmpeg `-ss/-to`, download trim.
- **youtube.py**: paste a URL, yt-dlp downloads to `/tmp`, optional audio extraction, download.

These import directly from `services/media/*` (and `services/media/youtube.py`). They don't go through the worker — pure local file operations.

## `theme.py`

Tiny module. `resolve_logo_path()` looks for the project logo PNG (multiple candidate paths) and returns the first that exists. Falls back gracefully.

`st_theme` is in requirements but currently used only for dark-mode detection; not deeply integrated.

## How the worker-down banner behaves

```python
def _check_worker():
    if "worker_alive" not in st.session_state:
        st.session_state.worker_alive = worker.is_alive()
    if not st.session_state.worker_alive:
        st.error(f"⚠️ Imagina worker is not reachable at `{WORKER_URL}` ...")
```

Cached in session state so we don't ping every rerun. To re-check after starting the worker:

```python
del st.session_state.worker_alive
st.rerun()
```

A "Recheck" button could be added; today the user reloads the tab.

## UI tab pattern (any tool)

For new tools, the canonical pattern:

```python
# ui/tabs/my_tool.py
import streamlit as st

def render() -> None:
    st.title(":wrench: My Tool")
    uploaded = st.file_uploader("Input", type=["mp4"])
    if not uploaded: return

    if st.button("Process"):
        with st.spinner("Processing..."):
            result_path = my_logic(uploaded)
        st.video(result_path)
        with open(result_path, "rb") as f:
            st.download_button("Download", data=f.read(), file_name="result.mp4")
```

Wire it into `app.py`:

```python
from ui.tabs import my_tool
tab1, tab2, ..., tab6 = st.tabs([..., "🔧 My Tool"])
with tab6: my_tool.render()
```

That's it. No registry, no worker integration unless your tool calls model endpoints.

## Future improvements

- **Typed `SessionState`** — a dataclass mirror with default values, instead of stringly-typed dict access scattered across the codebase. Catches typos statically.
- **Cancel button** for the cinematic flow — calls `worker.cancel_job(...)` for in-flight async jobs.
- **Per-scene regeneration progress** — currently the storyboard gallery's regen button blocks with `st.spinner`; could show structured progress like the main flow.
- **Settings persistence** — sidebar settings reset on session loss. Persisting to a small JSON file under `/tmp/imagina/session.json` would carry preferences across browser refreshes.
- **Worker-down recheck button** — one-click "I started it now" recheck instead of full page reload.
- **Tab routing via URL** — Streamlit's tabs don't deep-link; switching to `st.navigation` would let users bookmark "the YouTube tool".
- **Theme toggle** — `st_theme` integration for proper dark-mode support across all widgets (currently inconsistent).
- **Typed yaml view in the model picker** — show `requires_env`, `scene_duration`, `supported_dimensions` next to each option instead of just the name + tier badge.
- **Multi-user mode** — session state is per-tab; if we ever want to serve multiple users, switch to per-user namespaces (cookies, login).
