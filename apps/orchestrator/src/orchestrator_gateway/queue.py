"""Job queue abstractions."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Deque, Dict
from uuid import uuid4

from .models import JobStatus, TaskSpec


@dataclass
class QueueJob:
    """Internal representation of a job."""

    job_id: str
    task: TaskSpec
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: JobStatus = JobStatus.PENDING
    result: dict | None = None
    error: str | None = None


class BaseJobQueue(ABC):
    """Abstract base class for job queues."""

    @abstractmethod
    async def enqueue(self, task: TaskSpec) -> QueueJob:
        ...

    @abstractmethod
    async def dequeue(self) -> QueueJob:
        ...

    @abstractmethod
    async def ack(self, job: QueueJob, result: dict | None = None, error: str | None = None) -> None:
        ...

    @abstractmethod
    async def get(self, job_id: str) -> QueueJob | None:
        ...

    async def iter_pending(self) -> AsyncIterator[QueueJob]:
        raise NotImplementedError


class InMemoryJobQueue(BaseJobQueue):
    """A simple in-memory FIFO queue."""

    def __init__(self) -> None:
        self._queue: Deque[QueueJob] = deque()
        self._jobs: Dict[str, QueueJob] = {}
        self._queue_event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def enqueue(self, task: TaskSpec) -> QueueJob:
        async with self._lock:
            job = QueueJob(job_id=uuid4().hex, task=task)
            self._queue.append(job)
            self._jobs[job.job_id] = job
            self._queue_event.set()
            self._queue_event.clear()
            return job

    async def dequeue(self) -> QueueJob:
        while True:
            async with self._lock:
                if self._queue:
                    job = self._queue.popleft()
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.utcnow()
                    return job
            await self._queue_event.wait()

    async def ack(self, job: QueueJob, result: dict | None = None, error: str | None = None) -> None:
        async with self._lock:
            job.completed_at = datetime.utcnow()
            job.result = result
            job.error = error
            job.status = JobStatus.COMPLETED if error is None else JobStatus.FAILED

    async def get(self, job_id: str) -> QueueJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def iter_pending(self) -> AsyncIterator[QueueJob]:
        async with self._lock:
            for job in list(self._queue):
                yield job


async def create_queue(backend: str) -> BaseJobQueue:
    if backend == "memory":
        return InMemoryJobQueue()
    raise ValueError(f"Unsupported job queue backend: {backend}")
