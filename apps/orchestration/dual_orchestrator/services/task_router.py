from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

from ..models.task import Job, TaskRequest, TaskSpecification, new_job


@dataclass
class Skill:
    name: str
    description: str
    queue: str
    keywords: Iterable[str]

    def matches(self, utterance: str) -> bool:
        utterance_lower = utterance.lower()
        return any(re.search(rf"\b{re.escape(keyword.lower())}\b", utterance_lower) for keyword in self.keywords)


class SkillRegistry:
    def __init__(self, skills: Iterable[Skill]) -> None:
        self._skills = list(skills)

    def match(self, utterance: str) -> List[Skill]:
        matched = [skill for skill in self._skills if skill.matches(utterance)]
        if not matched:
            # default fallback - send to coding agent
            matched = [skill for skill in self._skills if skill.name == "coding"]
        return matched

    def to_dict(self) -> List[Dict[str, str]]:
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "queue": skill.queue,
                "keywords": list(skill.keywords),
            }
            for skill in self._skills
        ]


class TaskRouter:
    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def plan(self, request: TaskRequest) -> TaskSpecification:
        skills = self._registry.match(request.utterance)
        jobs: List[Job] = []
        trace_id = None
        for skill in skills:
            payload = {
                "utterance": request.utterance,
                "metadata": request.metadata,
                "context": request.context,
                "skill": skill.name,
            }
            job = new_job(task_type=skill.name, payload=payload, trace_id=trace_id)
            trace_id = job.trace_id
            jobs.append(job)
        return TaskSpecification.create(jobs)
