"""User-supplied SRT block → list of script dicts.

Re-exports `parse_script_scene_content` from services.llm.parser so callers
that think of "load my custom script" as a pipeline concern have a stable
import.
"""

from __future__ import annotations

from services.llm.parser import parse_script_scene_content, validate_script_data

__all__ = ["parse_script_scene_content", "validate_script_data"]
