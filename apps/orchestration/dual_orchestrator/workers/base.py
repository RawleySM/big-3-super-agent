from __future__ import annotations

import abc
import asyncio
import logging
from typing import Optional

from ..config import WorkerSettings
from ..models.task import Job
from ..queue.base import JobQueue
from ..services.orchestrator import WorkerContext

logger = logging.getLogger(__name__)


class Worker(abc.ABC):
    queue_name: str

    def __init__(self, settings: WorkerSettings, job_queue: JobQueue, context: WorkerContext) -> None:
        self.settings = settings
        self.job_queue = job_queue
        self.context = context
        self._tasks: list[asyncio.Task[None]] = []
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        for _ in range(self.settings.max_concurrency):
            task = asyncio.create_task(self._run_loop())
            self._tasks.append(task)

    async def stop(self) -> None:
        self._shutdown.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                job = await self.job_queue.dequeue(self.queue_name, timeout=self.settings.idle_sleep_seconds)
            except asyncio.TimeoutError:
                continue
            if not job:
                continue
            try:
                await self.context.mark_running(job)
                await self.handle(job)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.exception("Worker %s failed to process job %s", self.queue_name, job.job_id)
                await self.context.mark_failure(job, str(exc))

    @abc.abstractmethod
    async def handle(self, job: Job) -> None: ...
