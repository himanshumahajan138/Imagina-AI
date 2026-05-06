"""Artifact paths.

Phase 0 stub: just resolves paths under the system tempdir.
Phase 5+ swap target: S3 / Cloudflare R2 / MinIO adapter.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path


def temp_path(suffix: str = "") -> Path:
    name = f"{uuid.uuid4().hex}{suffix}"
    return Path(tempfile.gettempdir()) / name


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
