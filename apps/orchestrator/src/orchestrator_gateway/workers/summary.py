"""Summary worker producing simple responses."""
from __future__ import annotations

import asyncio
from typing import Any, Dict

import structlog

from ..models import TaskSpec
from ..observability.tracing import get_trace_writer, TraceSpan
from .base import Worker

_logger = structlog.get_logger(__name__)


class SummaryWorker(Worker):
    """Aggregates trace context or produces placeholder summaries."""

    def __init__(self) -> None:
        super().__init__(name="summary")

    async def execute(self, job_id: str, task: TaskSpec) -> Dict[str, Any]:
        trace_writer = get_trace_writer()
        await trace_writer.write(
            TraceSpan(
                trace_id=task.trace_id,
                job_id=job_id,
                component="worker",
                payload={"event": "summary.start", "utterance": task.utterance},
            )
        )
        await asyncio.sleep(0.05)
        result = {
            "summary": f"Summary placeholder for: {task.utterance}",
            "params": task.params,
        }
        await trace_writer.write(
            TraceSpan(
                trace_id=task.trace_id,
                job_id=job_id,
                component="worker",
                payload={"event": "summary.end", **result},
            )
        )
        _logger.info("summary generated", trace_id=task.trace_id, job_id=job_id)
        return result
