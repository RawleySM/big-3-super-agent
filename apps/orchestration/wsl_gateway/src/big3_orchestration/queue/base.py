from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class QueueItem:
    job_id: str
    payload: Dict[str, Any]
    skill: str


class JobQueue(abc.ABC):
    @abc.abstractmethod
    async def enqueue(self, item: QueueItem) -> None:
        ...

    @abc.abstractmethod
    async def dequeue(self, timeout: Optional[float] = None) -> QueueItem:
        ...

    @abc.abstractmethod
    async def ack(self, job_id: str) -> None:
        ...

    @abc.abstractmethod
    async def size(self) -> int:
        ...
