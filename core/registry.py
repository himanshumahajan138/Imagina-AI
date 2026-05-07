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


def is_stub(model_cfg: dict[str, Any]) -> bool:
    """`unavailable: true` in YAML marks a backend as a stub.

    Stubs always raise at runtime. Auto-pick skips them so the registry
    never silently lands on one when an env-satisfied tier is missing.
    """
    return bool(model_cfg.get("unavailable"))


def is_pickable(model_cfg: dict[str, Any]) -> bool:
    """A model is pickable if its env is satisfied AND it isn't a stub."""
    return env_satisfied(model_cfg) and not is_stub(model_cfg)


def available_models(modality: str) -> list[tuple[str, dict[str, Any]]]:
    """Return (model_id, cfg) pairs whose env requirements are satisfied.

    Includes stubs — `available_models` is also used to render the sidebar
    tier picker, where stubs need to remain selectable (with a warning).
    Use `is_pickable` if you want auto-pickable models only.
    """
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

    Selection rules:
      1. If `preferred` is set, its env is satisfied, and it isn't a stub
         → use it. (Explicit user pick of a stub still falls through here
         because the user accepted the trade-off when overriding from the
         sidebar; the backend will surface its own error at run time.)
      2. Else fall back api → cloud_oss → local across env-satisfied,
         non-stub models.
      3. Else raise BackendUnavailable with a hint about which env vars
         would unlock which backends.
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
        if is_pickable(cfg):
            by_tier[tier].append((mid, cfg))

    for tier in (Tier.API, Tier.CLOUD_OSS, Tier.LOCAL):
        if by_tier[tier]:
            return by_tier[tier][0]

    raise BackendUnavailable(_no_backend_message(modality, models))


def _no_backend_message(modality: str, models: dict[str, dict[str, Any]]) -> str:
    """Friendly diagnostic for 'no auto-pickable backend' state."""
    by_tier: dict[str, list[str]] = {"api": [], "cloud_oss": [], "local": []}
    needed: dict[str, list[str]] = {}
    for mid, cfg in models.items():
        tier = cfg.get("tier", "?")
        if is_stub(cfg):
            continue  # don't even mention stubs — they're not a real option
        by_tier.setdefault(tier, []).append(mid)
        req = cfg.get("requires_env")
        if req:
            keys = [req] if isinstance(req, str) else list(req)
            for k in keys:
                needed.setdefault(k, []).append(mid)

    lines = [f"No backend available for '{modality}'."]
    if needed:
        lines.append("Set one of these env vars to unlock a backend:")
        for env_var, mids in needed.items():
            lines.append(f"  • {env_var}  → {', '.join(mids)}")
    else:
        lines.append("Every backend is either a stub or has no env requirement.")
    return " ".join(lines) if len(lines) == 1 else "\n".join(lines)


def supported_dimensions(modality: str, model_id: str) -> list[str] | None:
    """Dimensions a model declares (None = no restriction = all DIMENSIONS).

    Stored in `models.yaml` as `supported_dimensions: ["1024x1024", ...]`.
    """
    cfg = list_models(modality).get(model_id, {})
    return cfg.get("supported_dimensions")


def common_dimensions(modalities: list[str]) -> list[str]:
    """Intersection of supported_dimensions across the active model in each modality.

    "Active" = `session_preferred` if set + env-satisfied, else the registry's
    auto-pick (api → cloud_oss → local). Modalities that fail to resolve a
    backend are skipped rather than collapsing the intersection.

    Returns dim values (e.g. "1024x1024") in the order they appear in
    `core.config.DIMENSIONS` so the sidebar dropdown stays stable.
    """
    from core.config import DIMENSIONS

    all_dims = list(DIMENSIONS.values())
    constraints: list[set[str]] = []
    for modality in modalities:
        try:
            _, cfg = pick_model(modality, preferred=session_preferred(modality))
        except Exception:
            continue
        sup = cfg.get("supported_dimensions")
        if sup:
            constraints.append(set(sup))

    if not constraints:
        return all_dims
    allowed = set.intersection(*constraints)
    return [d for d in all_dims if d in allowed]


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
