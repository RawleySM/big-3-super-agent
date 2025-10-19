"""Coding agent worker."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

import structlog

from ..config import get_settings
from ..models import TaskSpec
from ..observability.tracing import get_trace_writer, TraceSpan
from .base import Worker

_logger = structlog.get_logger(__name__)


class CodingAgentWorker(Worker):
    """Delegates code-generation tasks to an external coding agent."""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(name="coding")
        self.provider = settings.coding_agent_provider
        self.model = settings.coding_agent_model

    async def execute(self, job_id: str, task: TaskSpec) -> Dict[str, Any]:
        trace_writer = get_trace_writer()
        payload = {
            "utterance": task.utterance,
            "params": task.params,
            "provider": self.provider,
            "model": self.model,
        }
        await trace_writer.write(TraceSpan(trace_id=task.trace_id, job_id=job_id, component="worker", payload={"event": "coding.start", **payload}))
        # Placeholder: integrate with Anthropic Claude or other SDK.
        # We simulate asynchronous work and return a structured response.
        await asyncio.sleep(0.1)
        repo_root = Path(__file__).resolve().parents[4]
        result = {
            "provider": self.provider,
            "model": self.model,
            "summary": "Generated patch proposal (mock).",
            "repo_root": str(repo_root),
            "params": task.params,
        }
        await trace_writer.write(TraceSpan(trace_id=task.trace_id, job_id=job_id, component="worker", payload={"event": "coding.end", **result}))
        _logger.info("coding task completed", trace_id=task.trace_id, job_id=job_id, result=result)
        return result
