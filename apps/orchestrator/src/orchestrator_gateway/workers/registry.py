"""Registry for worker implementations."""
from __future__ import annotations

from typing import Dict

from ..models import TaskIntent
from .base import Worker
from .browser import BrowserAgentWorker
from .coding import CodingAgentWorker
from .summary import SummaryWorker


class WorkerRegistry:
    """Maintains mapping between intents and workers."""

    def __init__(self) -> None:
        self._workers: Dict[TaskIntent, Worker] = {
            TaskIntent.CODE: CodingAgentWorker(),
            TaskIntent.BROWSER: BrowserAgentWorker(),
            TaskIntent.SUMMARY: SummaryWorker(),
        }

    def get(self, intent: TaskIntent) -> Worker:
        if intent not in self._workers:
            raise KeyError(f"No worker registered for intent: {intent}")
        return self._workers[intent]

    def register(self, intent: TaskIntent, worker: Worker) -> None:
        self._workers[intent] = worker


_registry: WorkerRegistry | None = None


def get_worker_registry() -> WorkerRegistry:
    global _registry
    if _registry is None:
        _registry = WorkerRegistry()
    return _registry
