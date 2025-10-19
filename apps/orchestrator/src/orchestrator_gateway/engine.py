"""Task execution engine."""
from __future__ import annotations

import asyncio
import contextlib

import structlog

from .models import JobResult
from .observability.metrics import get_metrics_registry
from .observability.tracing import traced_span
from .queue import BaseJobQueue, QueueJob
from .workers.registry import get_worker_registry

_logger = structlog.get_logger(__name__)


class ExecutionEngine:
    """Consumes tasks from the queue and delegates to workers."""

    def __init__(self, queue: BaseJobQueue) -> None:
        self.queue = queue
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run_loop(self) -> None:
        registry = get_worker_registry()
        metrics = get_metrics_registry()
        while self._running:
            job = await self.queue.dequeue()
            worker = registry.get(job.task.intent)
            await metrics.incr("orchestrator_jobs_total")
            _logger.info("job dequeued", job_id=job.job_id, intent=job.task.intent.value)
            async with traced_span(job.task.trace_id, job.job_id, "worker", {"intent": job.task.intent.value}):
                try:
                    result = await worker.execute(job.job_id, job.task)
                    await self.queue.ack(job, result=result)
                    await metrics.incr("orchestrator_jobs_completed_total")
                except Exception as exc:  # pragma: no cover - logging path
                    await self.queue.ack(job, error=str(exc))
                    await metrics.incr("orchestrator_jobs_failed_total")
                    _logger.exception("job failed", job_id=job.job_id, error=str(exc))

    async def snapshot(self, job: QueueJob) -> JobResult:
        return JobResult(
            job_id=job.job_id,
            trace_id=job.task.trace_id,
            status=job.status,
            intent=job.task.intent,
            submitted_at=job.submitted_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            result=job.result,
            error=job.error,
        )
