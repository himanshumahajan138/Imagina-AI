"""Async job dispatch for long-running generations (video/lipsync).

Phase 0 stub. The synchronous in-process version below is enough for
Streamlit single-user mode. Production swap-in target: Redis + RQ or
Celery, with the same `submit()` / `result()` interface.
"""

from __future__ import annotations

import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class JobQueue:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Future] = {}

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        job_id = uuid.uuid4().hex
        self._jobs[job_id] = self._executor.submit(fn, *args, **kwargs)
        return job_id

    def is_done(self, job_id: str) -> bool:
        return self._jobs[job_id].done()

    def result(self, job_id: str, timeout: float | None = None) -> Any:
        return self._jobs[job_id].result(timeout=timeout)


_queue: JobQueue | None = None


def get_queue() -> JobQueue:
    global _queue
    if _queue is None:
        _queue = JobQueue()
    return _queue
