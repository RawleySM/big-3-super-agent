"""Browser automation worker."""
from __future__ import annotations

import asyncio
from typing import Any, Dict

import structlog

from ..config import get_settings
from ..models import TaskSpec
from ..observability.tracing import get_trace_writer, TraceSpan
from .base import Worker

_logger = structlog.get_logger(__name__)


class BrowserAgentWorker(Worker):
    """Executes browser automation tasks via Gemini Computer Use API or Playwright."""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(name="browser")
        self.provider = settings.browser_agent_provider
        self.model = settings.browser_agent_model

    async def execute(self, job_id: str, task: TaskSpec) -> Dict[str, Any]:
        trace_writer = get_trace_writer()
        payload = {
            "utterance": task.utterance,
            "params": task.params,
            "provider": self.provider,
            "model": self.model,
        }
        await trace_writer.write(TraceSpan(trace_id=task.trace_id, job_id=job_id, component="worker", payload={"event": "browser.start", **payload}))
        # Placeholder for real browser automation integration.
        await asyncio.sleep(0.2)
        actions = task.params.get("actions", [])
        result = {
            "provider": self.provider,
            "model": self.model,
            "actions": actions,
            "summary": "Executed browser automation script (mock).",
        }
        await trace_writer.write(TraceSpan(trace_id=task.trace_id, job_id=job_id, component="worker", payload={"event": "browser.end", **result}))
        _logger.info("browser task completed", trace_id=task.trace_id, job_id=job_id, result=result)
        return result
