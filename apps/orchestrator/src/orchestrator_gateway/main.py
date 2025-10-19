"""FastAPI entrypoint for orchestration gateway."""
from __future__ import annotations

import json
from datetime import datetime

import structlog
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import OrchestratorSettings, get_settings
from .engine import ExecutionEngine
from .models import JobResult, TaskRequest, TaskResponse
from .observability.metrics import get_metrics_registry
from .queue import BaseJobQueue, create_queue
from .routing import build_task_spec

_logger = structlog.get_logger(__name__)

app = FastAPI(title="Big-3 Orchestration Gateway")


async def get_queue(settings: OrchestratorSettings = Depends(get_settings)) -> BaseJobQueue:
    if not hasattr(app.state, "queue"):
        app.state.queue = await create_queue(settings.job_queue_backend)
    return app.state.queue


async def get_engine(queue: BaseJobQueue = Depends(get_queue)) -> ExecutionEngine:
    if not hasattr(app.state, "engine"):
        app.state.engine = ExecutionEngine(queue)
        await app.state.engine.start()
    return app.state.engine


@app.on_event("startup")
async def on_startup() -> None:
    settings = get_settings()
    structlog.configure(processors=[structlog.processors.JSONRenderer()])
    queue = await create_queue(settings.job_queue_backend)
    engine = ExecutionEngine(queue)
    await engine.start()
    app.state.queue = queue
    app.state.engine = engine
    _logger.info("gateway started", settings=settings.model_dump())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    engine: ExecutionEngine | None = getattr(app.state, "engine", None)
    if engine:
        await engine.stop()
    _logger.info("gateway stopped")


@app.post("/task", response_model=TaskResponse)
async def submit_task(request: TaskRequest, engine: ExecutionEngine = Depends(get_engine)) -> TaskResponse:
    queue: BaseJobQueue = engine.queue
    task_spec = build_task_spec(request.utterance, request.context)
    job = await queue.enqueue(task_spec)
    _logger.info("task submitted", job_id=job.job_id, trace_id=task_spec.trace_id, intent=task_spec.intent.value)
    response = TaskResponse(job_id=job.job_id, trace_id=task_spec.trace_id, accepted_at=datetime.utcnow())
    return response


@app.get("/status/{job_id}", response_model=JobResult)
async def get_status(job_id: str, engine: ExecutionEngine = Depends(get_engine)) -> JobResult:
    queue: BaseJobQueue = engine.queue
    job = await queue.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return await engine.snapshot(job)


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    registry = get_metrics_registry()
    export = await registry.export()
    return PlainTextResponse(export)


@app.get("/traces")
async def traces(limit: int = 100) -> JSONResponse:
    trace_path = get_settings().trace_log_path
    if not trace_path.exists():
        return JSONResponse([])
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    records = lines[-limit:]
    return JSONResponse([json.loads(line) for line in records])
