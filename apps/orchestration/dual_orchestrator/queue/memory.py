from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, Optional

from .base import JobQueue
from ..models.task import Job


class InMemoryJobQueue(JobQueue):
    """Simple in-memory asyncio job queue implementation."""

    def __init__(self) -> None:
        self._queues: Dict[str, Deque[Job]] = defaultdict(deque)
        self._condition = asyncio.Condition()

    async def enqueue(self, queue_name: str, job: Job) -> None:
        async with self._condition:
            self._queues[queue_name].append(job)
            self._condition.notify_all()

    async def dequeue(self, queue_name: str, timeout: Optional[float] = None) -> Optional[Job]:
        async with self._condition:
            if timeout is not None:
                await asyncio.wait_for(self._condition.wait_for(lambda: bool(self._queues[queue_name])), timeout)
            else:
                await self._condition.wait_for(lambda: bool(self._queues[queue_name]))
            return self._queues[queue_name].popleft()

    async def requeue(self, queue_name: str, job: Job) -> None:
        async with self._condition:
            self._queues[queue_name].appendleft(job)
            self._condition.notify_all()

    async def ack(self, queue_name: str, job: Job) -> None:  # pragma: no cover - noop for memory queue
        return None

    def queues(self) -> Iterable[str]:
        return list(self._queues.keys())
