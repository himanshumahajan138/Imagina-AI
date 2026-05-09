"""Load/unload heavy local models so we don't blow past the RAM ceiling.

The manager is process-wide; the worker daemon owns the only instance.
Streamlit reruns don't touch this — the worker keeps weights resident
across requests.

Eviction is *active*: callers can declare a phase boundary (e.g. "image
gen done, free SDXL") via `evict_modality()`, and the manager `del`s the
object, runs `gc.collect()`, and drains the MPS/CUDA/MLX GPU caches so
memory actually returns to the OS. Without the cache drain, `del` alone
leaves PyTorch holding the GPU allocator pool — eviction would be
purely cosmetic.
"""

from __future__ import annotations

import gc
from collections import OrderedDict
from threading import Lock
from typing import Any, Callable

from core.logger import logger


# Maps modality name → cache-key prefix used by that modality's backends.
# Backends create their cache keys with these prefixes (see e.g.
# services/llm/backends/mlx_local.py: f"mlx::{hf_id}"). Keep in sync.
# Prefixes are modality-specific so eviction can target one backend
# without collateral damage — e.g. SDXL (image) and LTX (video) both use
# diffusers but live under separate namespaces.
_MODALITY_PREFIXES: dict[str, tuple[str, ...]] = {
    "llm": ("mlx::",),
    "image": ("diffusers::",),
    "tts": ("kokoro::",),
    "video": ("ltx::",),
    "lipsync": ("wav2lip::",),
}


def _drain_gpu_caches() -> None:
    """Best-effort: tell PyTorch / MLX to release pooled GPU allocations.

    Without this, `del model` returns the Python-side reference but the
    GPU allocator keeps its arena, so RSS doesn't drop. Lazy imports so
    the manager works even when neither runtime is installed.
    """
    try:
        import torch  # type: ignore[import-not-found]
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[model_manager] torch cache drain failed: {e}")

    try:
        import mlx.core as mx  # type: ignore[import-not-found]
        # Newer MLX renamed `mx.metal.clear_cache` → `mx.clear_cache`.
        # Try the new name first, fall back to the legacy one.
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[model_manager] mlx cache drain failed: {e}")


class ModelManager:
    def __init__(self, max_resident_gb: float = 12.0) -> None:
        self._max_gb = max_resident_gb
        self._loaded: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = Lock()

    def get(self, model_id: str, loader: Callable[[], Any], cost_gb: float) -> Any:
        with self._lock:
            if model_id in self._loaded:
                self._loaded.move_to_end(model_id)
                return self._loaded[model_id][0]
            self._evict_until_fits_locked(cost_gb)
            obj = loader()
            self._loaded[model_id] = (obj, cost_gb)
            logger.info(f"[model_manager] loaded {model_id} ({cost_gb:.1f} GB)")
            return obj

    def evict(self, key: str) -> bool:
        """Evict one specific cache key. Returns True iff something was evicted.

        Performs the GPU cache drain after release so the kernel actually
        gets memory back, not just the Python interpreter.
        """
        with self._lock:
            entry = self._loaded.pop(key, None)
        if entry is None:
            return False
        obj, cost = entry
        del obj
        gc.collect()
        _drain_gpu_caches()
        logger.info(f"[model_manager] evicted {key} ({cost:.1f} GB freed)")
        return True

    def evict_modality(self, modality: str) -> int:
        """Evict every cached model belonging to a modality. Returns count."""
        prefixes = _MODALITY_PREFIXES.get(modality, ())
        if not prefixes:
            return 0
        with self._lock:
            keys = [k for k in self._loaded if any(k.startswith(p) for p in prefixes)]
        return sum(1 for k in keys if self.evict(k))

    def evict_all(self) -> int:
        with self._lock:
            n = len(self._loaded)
            self._loaded.clear()
        gc.collect()
        _drain_gpu_caches()
        if n:
            logger.info(f"[model_manager] evicted all ({n} models)")
        return n

    def resident(self) -> dict[str, float]:
        """Snapshot of currently-resident keys → declared cost in GB."""
        with self._lock:
            return {k: c for k, (_, c) in self._loaded.items()}

    def _evict_until_fits_locked(self, incoming_gb: float) -> None:
        """LRU evict to fit. Caller must hold the lock."""
        current = sum(c for _, c in self._loaded.values())
        evicted: list[str] = []
        while current + incoming_gb > self._max_gb and self._loaded:
            key, (obj, cost) = self._loaded.popitem(last=False)
            del obj
            evicted.append(key)
            current -= cost
        if evicted:
            gc.collect()
            _drain_gpu_caches()
            logger.info(
                f"[model_manager] LRU evicted {evicted} to fit incoming {incoming_gb:.1f} GB"
            )


_manager: ModelManager | None = None


def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
