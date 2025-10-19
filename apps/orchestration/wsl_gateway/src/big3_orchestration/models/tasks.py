from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SkillName(str, Enum):
    CODING = "coding"
    BROWSER = "browser"
    SUMMARISE = "summarise"


class TranscriptFragment(BaseModel):
    text: str
    timestamp: float


class TaskSpecification(BaseModel):
    utterance: str
    intent: SkillName
    payload: Dict[str, Any] = Field(default_factory=dict)
    fragments: List[TranscriptFragment] = Field(default_factory=list)


class TaskIn(BaseModel):
    utterance: str
    fragments: List[TranscriptFragment] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskOut(BaseModel):
    job_id: str
    intent: SkillName
    accepted: bool = True


class StatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
