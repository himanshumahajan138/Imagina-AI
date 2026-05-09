"""Shared helpers for Replicate-backed (tier 2) backends.

Auth is handled by `replicate` itself via the `REPLICATE_API_TOKEN` env var
— we don't pass it explicitly. This module just centralises the boring
parts: presence check, output normalisation, file download.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import requests

from core.errors import BackendUnavailable, GenerationFailed
from core.logger import logger


def require_token() -> None:
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise BackendUnavailable(
            "REPLICATE_API_TOKEN not set; cloud-OSS backends are unreachable."
        )


def run(model_ref: str, input: dict[str, Any]) -> Any:
    """Thin wrapper around `replicate.run` with a clear error path.

    Replicate's Python lib is a soft dep — only required when a tier-2
    backend is actually used. Importing it lazily keeps the rest of the
    project runnable without it installed.
    """
    require_token()
    try:
        import replicate  # type: ignore[import-not-found]
    except ImportError as e:
        raise BackendUnavailable(
            "The `replicate` package is not installed. "
            "Add it to requirements.txt (`pip install replicate`)."
        ) from e

    try:
        return replicate.run(model_ref, input=input)
    except Exception as e:
        raise GenerationFailed(f"Replicate call to {model_ref} failed: {e}") from e


def download(url: str, out_path: Path, timeout: int = 600) -> Path:
    """Stream a URL to disk; returns the path written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading replicate output: {url} -> {out_path}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
    return out_path


def first_url(output: Any) -> str:
    """Replicate outputs come in many shapes — list[str], str, FileOutput, …

    Reduce to a single URL string. Raises GenerationFailed if no URL found.
    """
    if output is None:
        raise GenerationFailed("Replicate returned no output")

    candidate = output
    if isinstance(candidate, (list, tuple)):
        if not candidate:
            raise GenerationFailed("Replicate returned empty list")
        candidate = candidate[0]

    # FileOutput-style objects expose `.url` or are str-castable
    url = getattr(candidate, "url", None) or str(candidate)
    if not isinstance(url, str) or not url.startswith("http"):
        raise GenerationFailed(f"Replicate output is not a URL: {url!r}")
    return url


def join_text(output: Any) -> str:
    """LLM outputs come as iterables of token strings; join them."""
    if isinstance(output, str):
        return output
    if isinstance(output, Iterable):
        return "".join(str(x) for x in output)
    raise GenerationFailed(f"Cannot interpret Replicate LLM output: {type(output)!r}")
