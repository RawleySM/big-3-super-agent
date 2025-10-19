from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import make_asgi_app

from .config import OrchestratorSettings
from .models.task import Job, TaskRequest, TaskSpecification
from .observability.tracing import TraceLogger
from .queue.memory import InMemoryJobQueue
from .services.job_store import JobStore
from .services.orchestrator import Orchestrator, WorkerContext
from .services.task_router import TaskRouter
from .skills.loader import load_registry
from .workers.browser_worker import BrowserWorker
from .workers.coding_worker import CodingWorker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TranscriptPayload(BaseModel):
    text: str
    is_final: bool
    latency_ms: float
    timestamp: float


class TaskPayload(BaseModel):
    utterance: str
    metadata: Dict[str, object] | None = None
    context: Dict[str, object] | None = None


class JobResponse(BaseModel):
    job_id: str
    task_type: str
    status: str


class TraceResponse(BaseModel):
    trace_id: str
    jobs: List[JobResponse]


settings = OrchestratorSettings.from_env()
trace_logger = TraceLogger(settings.observability.trace_log_path)
job_queue = InMemoryJobQueue()
job_store = JobStore()
registry_path = settings.skill_registry_path
if not registry_path.is_absolute():
    registry_path = Path(__file__).resolve().parent / registry_path
skill_registry = load_registry(registry_path)
router = TaskRouter(skill_registry)
orchestrator = Orchestrator(job_queue, job_store, router, trace_logger)
worker_context = WorkerContext(orchestrator)
coding_worker = CodingWorker(settings.coding_worker, worker_context, job_queue)
browser_worker = BrowserWorker(settings.browser_worker, worker_context, job_queue)

app = FastAPI(title="Big-3 Super Agent Orchestration Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/metrics", make_asgi_app())


@app.on_event("startup")
async def startup_event() -> None:
    await coding_worker.start()
    await browser_worker.start()
    logger.info("Workers started", extra={"skills": skill_registry.to_dict()})


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await coding_worker.stop()
    await browser_worker.stop()


async def enqueue_specification(specification: TaskSpecification) -> None:
    await orchestrator.enqueue(specification)


@app.post("/task", response_model=List[JobResponse])
async def create_task(payload: TaskPayload, background_tasks: BackgroundTasks) -> List[JobResponse]:
    request = TaskRequest(
        utterance=payload.utterance,
        metadata=payload.metadata or {},
        context=payload.context or {},
    )
    specification = router.plan(request)
    background_tasks.add_task(enqueue_specification, specification)
    return [
        JobResponse(job_id=job.job_id, task_type=job.task_type, status=job.status.value)
        for job in specification.jobs
    ]


@app.get("/status/{job_id}")
async def job_status(job_id: str) -> Dict[str, object]:
    job = await orchestrator.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/trace/{trace_id}")
async def trace(trace_id: str) -> TraceResponse:
    jobs = await orchestrator.get_trace(trace_id)
    return TraceResponse(
        trace_id=trace_id,
        jobs=[JobResponse(job_id=job.job_id, task_type=job.task_type, status=job.status.value) for job in jobs],
    )


@app.get("/skills")
async def skills() -> List[Dict[str, object]]:
    return skill_registry.to_dict()


@app.post("/transcript")
async def ingest_transcript(payload: TranscriptPayload, background_tasks: BackgroundTasks) -> Dict[str, str]:
    if not payload.is_final:
        trace_logger.log_event("transcript", "partial", phase="partial", payload=payload.model_dump())
        return {"status": "partial_received"}

    request = TaskRequest(utterance=payload.text, metadata={"latency_ms": payload.latency_ms}, context={"timestamp": payload.timestamp})
    specification = router.plan(request)
    background_tasks.add_task(enqueue_specification, specification)
    return {"status": "accepted"}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def create_app() -> FastAPI:
    return app


__all__ = ["app", "create_app"]
