from __future__ import annotations

import asyncio
from collections import deque
from typing import Deque, Optional

from .base import JobQueue, QueueItem


class InMemoryJobQueue(JobQueue):
    def __init__(self) -> None:
        self._queue: Deque[QueueItem] = deque()
        self._condition = asyncio.Condition()

    async def enqueue(self, item: QueueItem) -> None:
        async with self._condition:
            self._queue.append(item)
            self._condition.notify()

    async def dequeue(self, timeout: Optional[float] = None) -> QueueItem:
        async with self._condition:
            if not self._queue:
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    raise
            return self._queue.popleft()

    async def ack(self, job_id: str) -> None:  # pragma: no cover - placeholder
        return None

    async def size(self) -> int:
        async with self._condition:
            return len(self._queue)
