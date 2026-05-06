"""Load/unload heavy local models so we don't blow past the RAM ceiling.

Phase 0 stub. Phase 4 (tier-1 OSS backends) wires this up properly.

Design: a process-wide manager that holds at most N models resident,
evicts least-recently-used when a new one is requested, and tracks
approximate RAM/VRAM cost from `configs/models.yaml`.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Callable


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
            self._evict_until_fits(cost_gb)
            obj = loader()
            self._loaded[model_id] = (obj, cost_gb)
            return obj

    def _evict_until_fits(self, incoming_gb: float) -> None:
        current = sum(c for _, c in self._loaded.values())
        while current + incoming_gb > self._max_gb and self._loaded:
            _, (_, cost) = self._loaded.popitem(last=False)
            current -= cost

    def evict_all(self) -> None:
        with self._lock:
            self._loaded.clear()


_manager: ModelManager | None = None


def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
