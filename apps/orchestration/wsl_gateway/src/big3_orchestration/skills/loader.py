from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..models.tasks import SkillName, TaskSpecification, TaskIn


@dataclass
class Skill:
    name: SkillName
    phrases: List[str]
    worker: str


class SkillRegistry:
    def __init__(self, skills: List[Skill]) -> None:
        self._skills = skills

    @classmethod
    def from_json(cls, path: Path) -> "SkillRegistry":
        data = json.loads(path.read_text())
        skills = [Skill(name=SkillName(item["name"]), phrases=item["phrases"], worker=item["worker"]) for item in data]
        return cls(skills)

    def match(self, utterance: str) -> Skill:
        utterance_lower = utterance.lower()
        for skill in self._skills:
            if any(phrase in utterance_lower for phrase in skill.phrases):
                return skill
        return self._skills[0]


