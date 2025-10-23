import asyncio

import pytest

from orchestrator_gateway.models import TaskIntent, TaskSpec
from orchestrator_gateway.queue import InMemoryJobQueue


@pytest.mark.asyncio
async def test_queue_lifecycle():
    queue = InMemoryJobQueue()
    task = TaskSpec(utterance="refactor the api", intent=TaskIntent.CODE)
    job = await queue.enqueue(task)
    assert job.status.name == "PENDING"

    async def worker():
        dequeued = await queue.dequeue()
        assert dequeued.job_id == job.job_id
        await queue.ack(dequeued, result={"ok": True})

    await asyncio.gather(worker())
    stored = await queue.get(job.job_id)
    assert stored is not None
    assert stored.status.name == "COMPLETED"
