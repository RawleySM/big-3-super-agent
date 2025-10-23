from __future__ import annotations

import abc
import asyncio
import json
import time
from typing import Any, Dict

import structlog

from ..logging.json_logger import emit_trace
from ..queue.base import JobQueue, QueueItem

logger = structlog.get_logger(__name__)


class BaseWorker(abc.ABC):
    def __init__(self, name: str, queue: JobQueue, trace_logger, orchestrator_callback, metrics_counter=None) -> None:
        self._name = name
        self._queue = queue
        self._trace_logger = trace_logger
        self._orchestrator_callback = orchestrator_callback
        self._running = False
        self._metrics_counter = metrics_counter

    async def start(self) -> None:
        self._running = True
        logger.info("worker.started", name=self._name)
        while self._running:
            try:
                item = await self._queue.dequeue(timeout=1.0)
            except asyncio.TimeoutError:
                continue
            start = time.perf_counter()
            try:
                result = await self.handle(item)
                status = "completed"
            except Exception as exc:
                result = {"error": str(exc)}
                status = "failed"
                logger.exception("worker.error", name=self._name, job_id=item.job_id)
            duration = time.perf_counter() - start
            emit_trace(
                self._trace_logger,
                event="worker.finished",
                worker=self._name,
                job_id=item.job_id,
                status=status,
                duration_ms=duration * 1000,
                result=result,
            )
            self._orchestrator_callback(item.job_id, status, result)
            if self._metrics_counter:
                self._metrics_counter.labels(worker=self._name, status=status).inc()

    async def stop(self) -> None:
        self._running = False

    @abc.abstractmethod
    async def handle(self, item: QueueItem) -> Dict[str, Any]:
        ...
