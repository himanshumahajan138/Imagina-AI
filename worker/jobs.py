"""Async job manager for long-running endpoints (video / lipsync).

In-process `ThreadPoolExecutor` is enough for single-user mode. Same shape
as `core/job_queue.py` so the production swap to Redis+RQ later is
mechanical.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from core.logger import logger

JobStatusName = Literal["queued", "running", "done", "error", "cancelled"]


@dataclass
class JobRecord:
    job_id: str
    status: JobStatusName = "queued"
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    future: Future | None = None


class JobManager:
    """Thread-pool-backed job tracker. Thread-safe."""

    def __init__(self, max_workers: int = 2, ttl_seconds: int = 3600) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        job_id = uuid.uuid4().hex
        record = JobRecord(job_id=job_id)

        def _run() -> None:
            record.status = "running"
            record.started_at = time.time()
            try:
                record.result = fn(*args, **kwargs)
                record.status = "done"
            except Exception as e:
                logger.exception(f"[job {job_id}] failed")
                record.error = f"{type(e).__name__}: {e}"
                record.status = "error"
            finally:
                record.finished_at = time.time()

        record.future = self._executor.submit(_run)
        with self._lock:
            self._jobs[job_id] = record
            self._purge_stale_locked()
        logger.info(f"[job {job_id}] submitted")
        return job_id

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Best-effort cancel. Only effective for jobs still queued."""
        with self._lock:
            rec = self._jobs.get(job_id)
        if rec is None or rec.future is None or rec.future.done():
            return False
        if rec.future.cancel():
            rec.status = "cancelled"
            rec.finished_at = time.time()
            return True
        return False

    def _purge_stale_locked(self) -> None:
        """Drop finished jobs older than TTL. Caller must hold the lock."""
        cutoff = time.time() - self._ttl
        stale = [jid for jid, r in self._jobs.items() if r.finished_at and r.finished_at < cutoff]
        for jid in stale:
            del self._jobs[jid]
        if stale:
            logger.info(f"[jobs] purged {len(stale)} stale records")


# Module-level singleton — one queue per worker process.
manager = JobManager()
