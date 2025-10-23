from __future__ import annotations

import logging
import os
from typing import Any, Dict

import aiohttp

from ..config import BrowserWorkerSettings
from ..models.task import Job
from ..services.orchestrator import WorkerContext
from ..workers.base import Worker

logger = logging.getLogger(__name__)


class BrowserWorker(Worker):
    queue_name = "browser"

    def __init__(self, settings: BrowserWorkerSettings, context: WorkerContext, job_queue) -> None:
        super().__init__(settings, job_queue, context)
        self._settings = settings

    async def handle(self, job: Job) -> None:
        instruction = job.payload.get("utterance", "")
        metadata = job.payload.get("metadata", {})
        context = job.payload.get("context", {})
        logger.info("Browser worker executing job", extra={"job_id": job.job_id, "instruction": instruction})
        response = await self._invoke_gemini(instruction, metadata=metadata, context=context)
        await self.context.mark_success(job, response["output"], token_count=response["token_count"], cost_usd=response["cost_usd"])

    async def _invoke_gemini(self, instruction: str, *, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not configured; returning stub response")
            return {
                "output": {
                    "message": "Gemini browser automation skipped - configure GEMINI_API_KEY",
                    "instruction": instruction,
                    "metadata": metadata,
                    "context": context,
                },
                "token_count": 0,
                "cost_usd": 0.0,
            }
        # Example invocation for Gemini Computer Use API placeholder
        payload = {
            "instructions": instruction,
            "metadata": metadata,
            "context": context,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post("https://generativelanguage.googleapis.com/v1beta/gemini:computerUse", json=payload, headers=headers, timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()
        output = data.get("result", {})
        token_count = data.get("usage", {}).get("total_tokens", 0)
        cost = token_count * 0.000004  # placeholder pricing
        return {
            "output": output,
            "token_count": token_count,
            "cost_usd": cost,
        }
