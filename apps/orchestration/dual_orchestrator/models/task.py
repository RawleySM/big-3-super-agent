from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True)
class JobResult:
    success: bool
    output: Dict[str, Any]
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    token_count: Optional[int] = None
    cost_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "token_count": self.token_count,
            "cost_usd": self.cost_usd,
        }


@dataclass(slots=True)
class Job:
    trace_id: str
    job_id: str
    task_type: str
    payload: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    result: Optional[JobResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "job_id": self.job_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass(slots=True)
class TaskRequest:
    utterance: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskSpecification:
    trace_id: str
    jobs: List[Job]

    @classmethod
    def create(cls, jobs: List[Job]) -> "TaskSpecification":
        if not jobs:
            raise ValueError("TaskSpecification requires at least one job")
        return cls(trace_id=jobs[0].trace_id, jobs=jobs)


def new_job(task_type: str, payload: Dict[str, Any], trace_id: Optional[str] = None) -> Job:
    trace = trace_id or uuid.uuid4().hex
    return Job(trace_id=trace, job_id=uuid.uuid4().hex, task_type=task_type, payload=payload)
