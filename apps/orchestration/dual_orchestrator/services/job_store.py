from __future__ import annotations

import asyncio
from typing import Dict, Optional

from ..models.task import Job


class JobStore:
    """In-memory job store tracking status and results."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def save(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.job_id] = job

    async def get(self, job_id: str) -> Optional[Job]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def all_for_trace(self, trace_id: str) -> Dict[str, Job]:
        async with self._lock:
            return {job_id: job for job_id, job in self._jobs.items() if job.trace_id == trace_id}
