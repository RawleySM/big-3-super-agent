import asyncio

import pytest

from big3_orchestration.queue.memory import InMemoryJobQueue
from big3_orchestration.queue.base import QueueItem


@pytest.mark.asyncio
async def test_enqueue_dequeue_roundtrip():
    queue = InMemoryJobQueue()
    item = QueueItem(job_id="1", payload={"foo": "bar"}, skill="coding")
    await queue.enqueue(item)
    result = await queue.dequeue(timeout=0.1)
    assert result.job_id == "1"
    assert result.payload["foo"] == "bar"
