"""Natural language to task routing."""
from __future__ import annotations

import re
from typing import Dict

from .models import TaskIntent, TaskSpec


INTENT_PATTERNS: Dict[TaskIntent, list[re.Pattern[str]]] = {
    TaskIntent.CODE: [
        re.compile(r"\b(code|commit|pull request|refactor|implement)\b", re.I),
    ],
    TaskIntent.BROWSER: [
        re.compile(r"\b(browse|open|navigate|scrape|validate)\b", re.I),
    ],
    TaskIntent.SUMMARY: [
        re.compile(r"\b(summar(y|ise)|report|log review)\b", re.I),
    ],
}


def infer_intent(utterance: str) -> TaskIntent:
    for intent, patterns in INTENT_PATTERNS.items():
        if any(pattern.search(utterance) for pattern in patterns):
            return intent
    return TaskIntent.SUMMARY


def build_task_spec(utterance: str, context: dict | None = None) -> TaskSpec:
    intent = infer_intent(utterance)
    params = context or {}
    return TaskSpec(utterance=utterance, intent=intent, params=params)
