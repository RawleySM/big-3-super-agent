from __future__ import annotations

import uuid
from typing import Dict

import structlog

from ..logging.json_logger import emit_trace
from ..models.tasks import TaskIn, TaskOut, TaskSpecification
from ..queue.base import JobQueue, QueueItem
from ..skills.loader import SkillRegistry

logger = structlog.get_logger(__name__)


class OrchestratorService:
    def __init__(self, queue: JobQueue, skills: SkillRegistry, trace_logger) -> None:
        self._queue = queue
        self._skills = skills
        self._trace_logger = trace_logger
        self._status: Dict[str, Dict[str, object]] = {}

    async def submit(self, task: TaskIn) -> TaskOut:
        job_id = uuid.uuid4().hex
        skill = self._skills.match(task.utterance)
        spec = TaskSpecification(utterance=task.utterance, intent=skill.name, fragments=task.fragments, payload=task.metadata)
        queue_item = QueueItem(job_id=job_id, payload=spec.model_dump(), skill=skill.worker)
        await self._queue.enqueue(queue_item)
        self._status[job_id] = {"status": "queued", "result": None}
        emit_trace(
            self._trace_logger,
            event="task_submitted",
            job_id=job_id,
            intent=spec.intent,
            metadata=task.metadata,
            fragments=[fragment.model_dump() for fragment in task.fragments],
        )
        logger.info("task.submitted", job_id=job_id, intent=spec.intent)
        return TaskOut(job_id=job_id, intent=spec.intent)

    def update_status(self, job_id: str, status: str, result: Dict[str, object] | None = None) -> None:
        self._status[job_id] = {"status": status, "result": result}
        emit_trace(self._trace_logger, event="task_status", job_id=job_id, status=status, result=result)

    def get_status(self, job_id: str) -> Dict[str, object]:
        return self._status.get(job_id, {"status": "unknown"})
