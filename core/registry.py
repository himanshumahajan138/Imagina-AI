"""Backend selection logic.

Reads `configs/models.yaml`, inspects environment for available keys,
and returns the right backend instance for a (modality, model_id) pair.

Selection rules when caller passes `tier="auto"` (or no preference):
    1. If user explicitly picked a model → use it (if its env is satisfied).
    2. Else prefer in order: api → cloud_oss → local
       (skipping any tier whose required env vars are not set).
    3. Else raise BackendUnavailable with a helpful message.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from core.errors import BackendUnavailable, ConfigError
from core.types import Tier

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"

_TIER_LABEL = {
    Tier.LOCAL: "local",
    Tier.CLOUD_OSS: "cloud OSS",
    Tier.API: "API",
}


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        raise ConfigError(f"Model registry not found at {_CONFIG_PATH}")
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f) or {}


def list_models(modality: str) -> dict[str, dict[str, Any]]:
    reg = load_registry()
    if modality not in reg:
        raise ConfigError(f"Unknown modality: {modality}")
    return reg[modality].get("models", {})


def env_satisfied(model_cfg: dict[str, Any]) -> bool:
    required = model_cfg.get("requires_env")
    if not required:
        return True
    if isinstance(required, str):
        required = [required]
    return all(os.getenv(k) for k in required)


def available_models(modality: str) -> list[tuple[str, dict[str, Any]]]:
    """Return (model_id, cfg) pairs whose env requirements are satisfied."""
    return [
        (mid, cfg)
        for mid, cfg in list_models(modality).items()
        if env_satisfied(cfg)
    ]


def label_for(modality: str, model_id: str) -> str:
    """Return a friendly display label like 'gpt-4 (API)'."""
    cfg = list_models(modality).get(model_id, {})
    tier = cfg.get("tier", "?")
    tier_human = _TIER_LABEL.get(Tier(tier), tier) if tier in {t.value for t in Tier} else tier
    return f"{model_id} ({tier_human})"


def pick_model(
    modality: str,
    preferred: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (model_id, model_cfg) for the modality.

    If `preferred` is set and its env is satisfied, return it. Otherwise
    fall back to api → cloud_oss → local based on env availability.
    """
    models = list_models(modality)
    if preferred and preferred in models:
        cfg = models[preferred]
        if env_satisfied(cfg):
            return preferred, cfg

    by_tier: dict[Tier, list[tuple[str, dict[str, Any]]]] = {
        Tier.API: [],
        Tier.CLOUD_OSS: [],
        Tier.LOCAL: [],
    }
    for mid, cfg in models.items():
        try:
            tier = Tier(cfg["tier"])
        except (KeyError, ValueError):
            continue
        if env_satisfied(cfg):
            by_tier[tier].append((mid, cfg))

    for tier in (Tier.API, Tier.CLOUD_OSS, Tier.LOCAL):
        if by_tier[tier]:
            return by_tier[tier][0]

    raise BackendUnavailable(
        f"No backend available for modality '{modality}'. "
        f"Configure one of: {list(models.keys())}"
    )


def session_preferred(modality: str) -> str | None:
    """Read the user's tier-picker override from Streamlit session state.

    Returns None when streamlit isn't loaded (e.g. unit tests) or the
    user hasn't set an override.
    """
    try:
        import streamlit as st
    except ImportError:
        return None
    prefs = getattr(st, "session_state", {}).get("preferred_models", {}) if hasattr(st, "session_state") else {}
    val = prefs.get(modality)
    return val or None
