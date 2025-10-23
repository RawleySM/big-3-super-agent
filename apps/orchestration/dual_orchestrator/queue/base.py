from __future__ import annotations

import abc
from typing import Iterable, Optional

from ..models.task import Job


class JobQueue(abc.ABC):
    """Abstract job queue interface."""

    @abc.abstractmethod
    async def enqueue(self, queue_name: str, job: Job) -> None: ...

    @abc.abstractmethod
    async def dequeue(self, queue_name: str, timeout: Optional[float] = None) -> Optional[Job]: ...

    @abc.abstractmethod
    async def requeue(self, queue_name: str, job: Job) -> None: ...

    @abc.abstractmethod
    async def ack(self, queue_name: str, job: Job) -> None: ...

    @abc.abstractmethod
    def queues(self) -> Iterable[str]: ...
