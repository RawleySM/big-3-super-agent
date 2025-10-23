"""Pydantic models used by the orchestrator API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskIntent(str, Enum):
    CODE = "code"
    BROWSER = "browser"
    SUMMARY = "summary"


class TaskSpec(BaseModel):
    """Structured specification of a task created from natural language."""

    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    utterance: str
    intent: TaskIntent
    params: dict[str, Any] = Field(default_factory=dict)


class TaskRequest(BaseModel):
    """Request body for `/task` endpoint."""

    utterance: str
    context: Optional[dict[str, Any]] = None


class TaskResponse(BaseModel):
    """Response returned after task submission."""

    job_id: str
    trace_id: str
    accepted_at: datetime


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResult(BaseModel):
    """Represents job status results."""

    job_id: str
    trace_id: str
    status: JobStatus
    intent: TaskIntent
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class TraceRecord(BaseModel):
    """Single trace entry persisted to JSONL and optional SQLite."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: str
    job_id: str
    component: Literal["gateway", "worker", "edge"]
    payload: dict[str, Any]
