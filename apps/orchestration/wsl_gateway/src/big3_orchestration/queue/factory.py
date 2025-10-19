from __future__ import annotations

from .base import JobQueue
from .memory import InMemoryJobQueue


def create_queue(backend: str) -> JobQueue:
    if backend == "memory":
        return InMemoryJobQueue()
    raise NotImplementedError(f"Unsupported queue backend: {backend}")
