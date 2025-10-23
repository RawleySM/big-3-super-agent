from __future__ import annotations

import pytest

from dual_orchestrator.models.task import TaskRequest
from dual_orchestrator.queue.memory import InMemoryJobQueue
from dual_orchestrator.services.job_store import JobStore
from dual_orchestrator.services.orchestrator import Orchestrator, WorkerContext
from dual_orchestrator.services.task_router import SkillRegistry, TaskRouter
from dual_orchestrator.skills.loader import default_skills
from dual_orchestrator.observability.tracing import TraceLogger


@pytest.mark.asyncio
async def test_task_router_fan_out(tmp_path):
    trace_log = tmp_path / "traces.jsonl"
    trace_logger = TraceLogger(trace_log)
    job_queue = InMemoryJobQueue()
    job_store = JobStore()
    router = TaskRouter(SkillRegistry(default_skills()))
    orchestrator = Orchestrator(job_queue, job_store, router, trace_logger)

    request = TaskRequest(utterance="Please refactor the code and validate the website")
    spec = await orchestrator.submit(request)
    assert len(spec.jobs) == 2
    queues = set(job.task_type for job in spec.jobs)
    assert queues == {"coding", "browser"}


@pytest.mark.asyncio
async def test_job_store_roundtrip(tmp_path):
    trace_log = tmp_path / "traces.jsonl"
    trace_logger = TraceLogger(trace_log)
    job_queue = InMemoryJobQueue()
    job_store = JobStore()
    router = TaskRouter(SkillRegistry(default_skills()))
    orchestrator = Orchestrator(job_queue, job_store, router, trace_logger)
    context = WorkerContext(orchestrator)

    request = TaskRequest(utterance="summarize the logs")
    spec = await orchestrator.submit(request)
    job = spec.jobs[0]
    await context.mark_running(job)
    await context.mark_success(job, {"message": "done"}, token_count=10, cost_usd=0.01)

    stored = await job_store.get(job.job_id)
    assert stored is not None
    assert stored.status.value == "succeeded"
    assert stored.result is not None
    assert stored.result.output["message"] == "done"
