from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from ..services.task_router import Skill, SkillRegistry


def load_registry(path: Path) -> SkillRegistry:
    if path.exists():
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        skills = [
            Skill(
                name=item["name"],
                description=item.get("description", ""),
                queue=item.get("queue", item["name"]),
                keywords=item.get("keywords", []),
            )
            for item in data
        ]
    else:
        skills = default_skills()
    return SkillRegistry(skills)


def default_skills() -> List[Skill]:
    return [
        Skill(
            name="coding",
            description="Code generation, refactoring, and repository maintenance tasks",
            queue="coding",
            keywords=["code", "refactor", "implement", "python", "typescript", "fix", "bug"],
        ),
        Skill(
            name="browser",
            description="Browser automation, validation, and data extraction",
            queue="browser",
            keywords=["browser", "website", "validate", "check", "screenshot", "playwright", "gemini"],
        ),
        Skill(
            name="summarize",
            description="Summarize transcripts, logs, and agent outputs",
            queue="coding",
            keywords=["summarize", "explain", "recap", "overview"],
        ),
    ]


__all__ = ["load_registry", "default_skills"]
