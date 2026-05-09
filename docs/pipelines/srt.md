# SRT Pipeline

[pipelines/srt.py](../../pipelines/srt.py) — a thin re-export shim, not a fully-fledged pipeline.

## What it is

```python
from services.llm.parser import parse_script_scene_content, validate_script_data

__all__ = ["parse_script_scene_content", "validate_script_data"]
```

That's the whole module. It exists so callers that conceptually think of "load my custom SRT-style script" as a *pipeline* concern have a stable import path:

```python
from pipelines.srt import parse_script_scene_content
```

…instead of having to know the underlying functions live in `services/llm/parser.py`.

## Why this is an "underdeveloped" pipeline

A real cinematic-equivalent SRT pipeline would:

1. Accept an SRT-style file (already-written script + scene + video_scene lines per beat).
2. Skip the LLM phase entirely.
3. Run audio + image + video + lipsync exactly like cinematic does.
4. Emit a final merged video.

Today the parsing half lives here, but the orchestration half is folded into `ui/tabs/cinematic.py:_task_handler("load_script", ...)`. When the user uploads a `.txt` file in the sidebar's "Upload Existing Script File" expander, that handler:

1. Reads the bytes via `st.session_state.script_file.read().decode("utf-8")`.
2. Calls `parse_script_scene_content` (re-exported from this module's home in `services/llm/parser.py`).
3. Wraps the parsed dicts in a DataFrame with `.assign(speed=, speaker=, custom_image=None)`.
4. Stores it in `st.session_state.script_df`.

From that point on, the cinematic flow takes over — the user can edit, regenerate scenes, etc. exactly like they would after LLM-generated script.

So functionally, "SRT load" is supported end-to-end; it's just spread across `services/llm/parser.py` (parsing) + `ui/tabs/cinematic.py` (loading) rather than being a single `pipelines/srt.py:run()` function.

## Format the parser expects

```
00:00:00,000 --> 00:00:08,000
[script]: "Spoken line in the chosen language."
[scene]: "English image-prompt paragraph for the still."
[video_scene]: "English video-prompt paragraph (camera moves, motion, mood)."

00:00:08,000 --> 00:00:16,000
[script]: "Next line of dialogue."
[scene]: "Next image prompt."
[video_scene]: "Next video direction."
```

Block-by-block parse, lenient on whitespace, logs warnings on malformed blocks rather than raising.

Validation rules (from `services/llm/parser.py:validate_script_data`):

- `start_time` / `end_time` must be SRT format and `start < end`.
- Adjacent rows must be contiguous (no gaps or overlaps).
- Required text fields must be non-empty.
- Max `end_time` cannot exceed the global duration slider.

The sidebar exposes the format in a help expander, with explicit examples.

## Future improvements

- **Promote `srt.py` to a real pipeline** — `pipelines/srt.py:run(text, **kwargs)` that does the full load → image → video → merge flow as a single call. Useful from a CLI or batch job.
- **Multi-format support** — SubRip (.srt), WebVTT (.vtt), TTML. Currently only the bespoke `[script]/[scene]/[video_scene]` block format works.
- **Validation error UX** — currently warnings get logged but malformed blocks are silently skipped. Surface them in the UI so the user can fix the input.
- **Round-trip via the editor** — let the user export the edited DataFrame back to the SRT-style format. Currently you can load but not save.
