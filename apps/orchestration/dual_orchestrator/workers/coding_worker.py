from __future__ import annotations

import logging
import os
from typing import Any, Dict

import aiohttp

from ..config import CodingWorkerSettings
from ..models.task import Job
from ..services.orchestrator import WorkerContext
from ..workers.base import Worker

logger = logging.getLogger(__name__)


class CodingWorker(Worker):
    queue_name = "coding"

    def __init__(self, settings: CodingWorkerSettings, context: WorkerContext, job_queue) -> None:
        super().__init__(settings, job_queue, context)
        self._settings = settings

    async def handle(self, job: Job) -> None:
        prompt = job.payload.get("utterance", "")
        metadata = job.payload.get("metadata", {})
        context = job.payload.get("context", {})
        logger.info("Coding worker executing job", extra={"job_id": job.job_id, "prompt": prompt})
        response = await self._invoke_claude(prompt, metadata=metadata, context=context)
        await self.context.mark_success(job, response["output"], token_count=response["token_count"], cost_usd=response["cost_usd"])

    async def _invoke_claude(self, prompt: str, *, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not configured; returning stub response")
            return {
                "output": {
                    "message": "Claude invocation skipped - configure ANTHROPIC_API_KEY to enable",
                    "prompt": prompt,
                    "metadata": metadata,
                    "context": context,
                },
                "token_count": 0,
                "cost_usd": 0.0,
            }
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self._settings.anthropic_model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()
        content = data.get("content", [])
        text = "\n".join(block.get("text", "") for block in content)
        usage = data.get("usage", {})
        tokens = usage.get("output_tokens", 0)
        cost = tokens * 0.000008  # placeholder pricing
        return {
            "output": {"message": text, "metadata": metadata, "context": context},
            "token_count": tokens,
            "cost_usd": cost,
        }
