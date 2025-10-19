from __future__ import annotations

import asyncio
import time
from typing import Dict, List

from ..models.task import Job, JobResult, JobStatus, TaskRequest, TaskSpecification
from ..observability.tracing import TraceLogger
from ..queue.base import JobQueue
from .job_store import JobStore
from .task_router import TaskRouter


class Orchestrator:
    def __init__(
        self,
        job_queue: JobQueue,
        job_store: JobStore,
        task_router: TaskRouter,
        trace_logger: TraceLogger,
    ) -> None:
        self._queue = job_queue
        self._store = job_store
        self._router = task_router
        self._trace_logger = trace_logger

    async def submit(self, request: TaskRequest) -> TaskSpecification:
        specification = self._router.plan(request)
        await self.enqueue(specification)
        return specification

    async def enqueue(self, specification: TaskSpecification) -> None:
        for job in specification.jobs:
            await self._store.save(job)
            await self._queue.enqueue(job.task_type, job)
            self._trace_logger.log_event(
                job.trace_id,
                job.job_id,
                phase="submitted",
                payload={"task_type": job.task_type, "payload": job.payload},
            )

    async def update_job_status(self, job: Job, status: JobStatus, result: JobResult | None = None) -> None:
        job.status = status
        if result:
            job.result = result
        await self._store.save(job)
        payload: Dict[str, object] = {"status": status.value}
        if result:
            payload.update(result.to_dict())
        self._trace_logger.log_event(job.trace_id, job.job_id, phase=status.value, payload=payload)

    async def get_job(self, job_id: str) -> Job | None:
        return await self._store.get(job_id)

    async def get_trace(self, trace_id: str) -> List[Job]:
        jobs = await self._store.all_for_trace(trace_id)
        return list(jobs.values())


class WorkerContext:
    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orchestrator = orchestrator

    async def mark_running(self, job: Job) -> None:
        await self._orchestrator.update_job_status(job, JobStatus.RUNNING)

    async def mark_success(self, job: Job, output: Dict[str, object], *, token_count: int | None = None, cost_usd: float | None = None) -> None:
        result = JobResult(success=True, output=output, token_count=token_count, cost_usd=cost_usd)
        result.completed_at = time.time()
        await self._orchestrator.update_job_status(job, JobStatus.SUCCEEDED, result)

    async def mark_failure(self, job: Job, error: str) -> None:
        result = JobResult(success=False, output={}, error=error)
        result.completed_at = time.time()
        await self._orchestrator.update_job_status(job, JobStatus.FAILED, result)
