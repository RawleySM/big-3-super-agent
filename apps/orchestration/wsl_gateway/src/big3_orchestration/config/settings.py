from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.wsl", env_file_encoding="utf-8", extra="allow")

    app_env: Literal["development", "production", "test"] = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    job_queue_backend: Literal["memory", "redis"] = "memory"
    trace_log_path: Path = Path("/tmp/big3_traces.jsonl")
    sqlite_path: Optional[Path] = Path("./orchestration_traces.db")
    prometheus_metrics_port: int = 9108
    claude_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
