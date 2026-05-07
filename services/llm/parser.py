"""Script parsers — model JSON output AND user-supplied SRT-style text."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from core.logger import logger
from core.types import ScriptBlock

_BLOCK_FIELDS = {"script", "scene", "video_scene", "start_time", "end_time"}


def row_to_block(row: dict[str, Any]) -> ScriptBlock:
    """Map an LLM-emitted row dict to a ScriptBlock dataclass.

    Unknown keys are preserved on `extra` so we don't lose model-specific
    metadata (e.g. confidence scores from a future provider).
    """
    known = {k: row.get(k, "") for k in _BLOCK_FIELDS}
    return ScriptBlock(**known)


def is_valid_srt_timestamp(ts: str) -> bool:
    return bool(re.match(r"^\d{2}:\d{2}:\d{2},\d{3}$", ts))


def parse_srt_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%H:%M:%S,%f")


def to_seconds(dt: datetime) -> float:
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000


def validate_script_data(script_data: list[dict], global_duration: int) -> list[str]:
    """Validate each row of script_data; return a list of error strings."""
    errors: list[str] = []
    max_end_seconds: float = 0

    for i, row in enumerate(script_data, start=1):
        start_time = row.get("start_time", "").strip()
        end_time = row.get("end_time", "").strip()

        if not is_valid_srt_timestamp(start_time):
            errors.append(
                f"Scene {i}: Invalid start_time format: '{start_time}'  \n"
                f"Please use format HH:MM:SS,mmm (e.g., 00:00:05,000)"
            )
        if not is_valid_srt_timestamp(end_time):
            errors.append(
                f"Scene {i}: Invalid end_time format: '{end_time}'  \n"
                f"Please use format HH:MM:SS,mmm (e.g., 00:00:10,450)"
            )

        if is_valid_srt_timestamp(start_time) and is_valid_srt_timestamp(end_time):
            start_dt = parse_srt_timestamp(start_time)
            end_dt = parse_srt_timestamp(end_time)
            if start_dt >= end_dt:
                errors.append(f"Scene {i}: start_time must be before end_time.")
            max_end_seconds = max(max_end_seconds, to_seconds(end_dt))

            if i > 1:
                prev_end = parse_srt_timestamp(script_data[i - 2]["end_time"])
                if prev_end != start_dt:
                    errors.append(
                        f"Scene {i-1} end_time `{script_data[i-2]['end_time']}` "
                        f"must exactly match Scene {i} start_time `{start_time}` "
                        "(no gaps or overlaps)."
                    )

        if not row.get("script") or not str(row["script"]).strip():
            errors.append(f"Scene {i}: Script cannot be empty.")
        if not row.get("scene") or not str(row["scene"]).strip():
            errors.append(f"Scene {i}: Scene cannot be empty.")
        if not row.get("video_scene") or not str(row["video_scene"]).strip():
            errors.append(f"Scene {i}: Video Scene cannot be empty.")

    if max_end_seconds > global_duration:
        formatted_duration = str(timedelta(seconds=global_duration))
        formatted_max = str(timedelta(seconds=max_end_seconds))
        errors.append(
            "⏱️ Total duration exceeds global limit.  \n"
            f"- Max end_time in script: `{formatted_max}`  \n"
            f"- Allowed global duration: `{formatted_duration}`"
        )

    return errors


def parse_script_scene_content(text: str) -> list[dict]:
    """Parse a block-structured SRT-style script with [script]/[scene]/[video_scene] tags."""
    blocks = re.split(r"\n\s*\n", text.strip())
    parsed: list[dict] = []

    for i, block in enumerate(blocks):
        lines = block.strip().splitlines()
        if not lines:
            continue

        timestamp_line = lines[0].strip()
        timestamp_match = re.match(
            r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
            r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})",
            timestamp_line,
        )
        if not timestamp_match:
            logger.warning(f"Invalid or missing timestamp in block #{i+1}. Skipping.")
            continue

        start_time = timestamp_match.group("start")
        end_time = timestamp_match.group("end")

        script = scene = video_scene = ""

        for line in lines[1:]:
            line = line.strip()
            if line.lower().startswith("[script]:"):
                m = re.match(r'\[script\]:\s*"(.*?)"\s*$', line, re.IGNORECASE)
                script = m.group(1).strip() if m else line.split(":", 1)[1].strip().strip('"')
            elif line.lower().startswith("[scene]:"):
                m = re.match(r'\[scene\]:\s*"(.*?)"\s*$', line, re.IGNORECASE)
                scene = m.group(1).strip() if m else line.split(":", 1)[1].strip().strip('"')
            elif line.lower().startswith("[video_scene]:"):
                m = re.match(r'\[video_scene\]:\s*"(.*?)"\s*$', line, re.IGNORECASE)
                video_scene = m.group(1).strip() if m else line.split(":", 1)[1].strip().strip('"')

        if not script:
            logger.warning(f"Missing [script] in block #{i+1}. Skipping.")
            continue
        if not scene:
            logger.warning(f"Missing [scene] in block #{i+1}. Skipping.")
            continue
        if not video_scene:
            logger.info(f"No [video_scene] provided in block #{i+1}. Leaving blank.")

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


# Back-compat re-exports (used by core/utils.py during the transition)
_is_valid_srt_timestamp = is_valid_srt_timestamp
_parse_srt_timestamp = parse_srt_timestamp
_to_seconds = to_seconds
