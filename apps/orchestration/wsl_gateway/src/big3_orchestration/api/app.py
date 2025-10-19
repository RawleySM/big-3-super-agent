from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest

from ..config.settings import get_settings
from ..logging.json_logger import configure_logging
from ..models.tasks import StatusResponse, TaskIn, TaskOut
from ..queue.factory import create_queue
from ..services.orchestrator import OrchestratorService
from ..skills.loader import SkillRegistry
from ..workers.browser import BrowserWorker
from ..workers.coding import CodingWorker

app = FastAPI(title="Big-3 Orchestration Gateway", version="0.1.0")

settings = get_settings()
trace_logger = configure_logging(settings.trace_log_path)
skill_registry = SkillRegistry.from_json(Path(__file__).resolve().parent.parent / "skills" / "registry.json")
queue = create_queue(settings.job_queue_backend)
orchestrator = OrchestratorService(queue, skill_registry, trace_logger)

registry = CollectorRegistry()
queue_depth_metric = Gauge("big3_queue_depth", "Number of queued jobs", registry=registry)
worker_completed_metric = Counter("big3_worker_completed", "Completed jobs", ["worker", "status"], registry=registry)


async def queue_depth_updater() -> None:
    while True:
        queue_depth_metric.set(await queue.size())
        await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event() -> None:
    asyncio.create_task(queue_depth_updater())
    asyncio.create_task(
        CodingWorker(queue, trace_logger, orchestrator.update_status, Path.cwd(), worker_completed_metric).start()
    )
    asyncio.create_task(
        BrowserWorker(queue, trace_logger, orchestrator.update_status, worker_completed_metric).start()
    )


@app.post("/task", response_model=TaskOut)
async def create_task(task: TaskIn) -> TaskOut:
    result = await orchestrator.submit(task)
    queue_depth_metric.set(await queue.size())
    return result


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str) -> StatusResponse:
    data = orchestrator.get_status(job_id)
    if data["status"] == "unknown":
        raise HTTPException(404, "job not found")
    return StatusResponse(job_id=job_id, status=data["status"], result=data.get("result"))


@app.post("/transcript", response_model=TaskOut)
async def transcript_endpoint(task: TaskIn) -> TaskOut:
    return await create_task(task)


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(registry), media_type="text/plain; version=0.0.4")
