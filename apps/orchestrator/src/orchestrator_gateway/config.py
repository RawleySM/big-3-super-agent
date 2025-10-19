"""Configuration utilities for the orchestrator gateway."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os


ENV_FILES = [
    Path(__file__).resolve().parents[3] / ".env.wsl",
    Path(__file__).resolve().parents[3] / ".env.local",
]


def load_env() -> None:
    """Load .env files in priority order."""
    for env_file in ENV_FILES:
        if env_file.exists():
            load_dotenv(env_file, override=False)


class OrchestratorSettings(BaseModel):
    """Runtime configuration for the orchestrator gateway."""

    fastapi_host: str = Field(default_factory=lambda: os.getenv("FASTAPI_HOST", "0.0.0.0"))
    fastapi_port: int = Field(default_factory=lambda: int(os.getenv("FASTAPI_PORT", "8000")))
    job_queue_backend: str = Field(default_factory=lambda: os.getenv("JOB_QUEUE_BACKEND", "memory"))
    trace_log_path: Path = Field(default_factory=lambda: Path(os.getenv("TRACE_LOG_PATH", "logs/orchestrator-traces.jsonl")))
    trace_db_path: Path = Field(default_factory=lambda: Path(os.getenv("TRACE_DB_PATH", "data/traces.sqlite")))
    coding_agent_provider: str = Field(default_factory=lambda: os.getenv("CODING_AGENT_PROVIDER", "anthropic"))
    coding_agent_model: str = Field(default_factory=lambda: os.getenv("CODING_AGENT_MODEL", "claude-3-5-sonnet"))
    browser_agent_provider: str = Field(default_factory=lambda: os.getenv("BROWSER_AGENT_PROVIDER", "gemini"))
    browser_agent_model: str = Field(default_factory=lambda: os.getenv("BROWSER_AGENT_MODEL", "gemini-1.5-pro"))
    default_trace_sample_rate: float = Field(default_factory=lambda: float(os.getenv("DEFAULT_TRACE_SAMPLE_RATE", "1.0")))

    class Config:
        arbitrary_types_allowed = True


@lru_cache(maxsize=1)
def get_settings() -> OrchestratorSettings:
    """Return cached orchestrator settings."""
    load_env()
    settings = OrchestratorSettings()
    settings.trace_log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.trace_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings


def settings_dict() -> dict[str, Any]:
    """Convenience helper for exporting settings to logs."""
    return get_settings().model_dump()
