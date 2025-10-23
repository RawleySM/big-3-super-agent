import asyncio
from pathlib import Path

import pytest

from big3_orchestration.logging.json_logger import configure_logging
from big3_orchestration.models.tasks import TaskIn
from big3_orchestration.queue.memory import InMemoryJobQueue
from big3_orchestration.services.orchestrator import OrchestratorService
from big3_orchestration.skills.loader import SkillRegistry


@pytest.mark.asyncio
async def test_orchestrator_submits_jobs(tmp_path: Path):
    registry = SkillRegistry.from_json(Path(__file__).resolve().parents[1] / "src" / "big3_orchestration" / "skills" / "registry.json")
    queue = InMemoryJobQueue()
    trace_logger = configure_logging(tmp_path / "trace.jsonl")
    orchestrator = OrchestratorService(queue, registry, trace_logger)

    task = TaskIn(utterance="Please generate code for me", fragments=[])
    result = await orchestrator.submit(task)
    assert result.intent.value == "coding"
    queued = await queue.dequeue(timeout=0.1)
    assert queued.payload["utterance"] == task.utterance
