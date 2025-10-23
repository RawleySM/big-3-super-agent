"""Base classes for worker processes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..models import TaskSpec


class Worker(ABC):
    """Abstract worker capable of executing a task."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    async def execute(self, job_id: str, task: TaskSpec) -> Dict[str, Any]:
        ...
