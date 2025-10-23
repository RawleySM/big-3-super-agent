from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv


def _load_env_file(default: str) -> None:
    """Load environment variables from the provided env file if it exists."""
    env_file = Path(os.environ.get("DUAL_ENV_FILE", default))
    if env_file.exists():
        load_dotenv(env_file)


class WindowsEdgeSettings(BaseModel):
    """Settings for the Windows real-time voice edge."""

    realtime_api_url: str = Field(default="wss://api.openai.com/v1/realtime")
    realtime_model: str = Field(default="gpt-4o-realtime-preview")
    local_gateway_url: str = Field(default="http://127.0.0.1:8080")
    microphone_sample_rate: int = Field(default=16_000)
    frame_ms: int = Field(default=30)
    vad_energy_threshold: float = Field(default=0.01)
    reconnect_backoff_seconds: int = Field(default=5)

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "WindowsEdgeSettings":
        if env_file:
            _load_env_file(env_file)
        else:
            _load_env_file(".env.local")
        values: Dict[str, Any] = {}
        if gateway := os.environ.get("WSL_GATEWAY_URL"):
            values["local_gateway_url"] = gateway
        if frame := os.environ.get("FRAME_MS"):
            values["frame_ms"] = int(frame)
        if vad := os.environ.get("VAD_ENERGY_THRESHOLD"):
            values["vad_energy_threshold"] = float(vad)
        if backoff := os.environ.get("EDGE_RECONNECT_BACKOFF"):
            values["reconnect_backoff_seconds"] = int(backoff)
        if model := os.environ.get("REALTIME_MODEL"):
            values["realtime_model"] = model
        return cls.model_validate(values)


class QueueSettings(BaseModel):
    backend: str = Field(default="memory")
    visibility_timeout_seconds: int = Field(default=900)
    reservation_timeout_seconds: int = Field(default=60)


class ObservabilitySettings(BaseModel):
    trace_log_path: Path = Field(default=Path("logs/traces.jsonl"))
    metrics_host: str = Field(default="0.0.0.0")
    metrics_port: int = Field(default=9001)
    sqlite_path: Path = Field(default=Path("logs/traces.db"))


class WorkerSettings(BaseModel):
    max_concurrency: int = Field(default=1)
    idle_sleep_seconds: float = Field(default=0.25)


class CodingWorkerSettings(WorkerSettings):
    anthropic_model: str = Field(default="claude-3-5-sonnet-latest")
    working_directory: Path = Field(default=Path("."))


class BrowserWorkerSettings(WorkerSettings):
    headless: bool = Field(default=True)
    default_timeout_ms: int = Field(default=20_000)


class OrchestratorSettings(BaseModel):
    """Settings for the WSL orchestration layer."""

    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    gateway_host: str = Field(default="0.0.0.0")
    gateway_port: int = Field(default=8080)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    coding_worker: CodingWorkerSettings = Field(default_factory=CodingWorkerSettings)
    browser_worker: BrowserWorkerSettings = Field(default_factory=BrowserWorkerSettings)
    skill_registry_path: Path = Field(default=Path("skills/registry.json"))

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "OrchestratorSettings":
        if env_file:
            _load_env_file(env_file)
        else:
            _load_env_file(".env.wsl")
        values: Dict[str, Any] = {}
        if host := os.environ.get("WSL_GATEWAY_HOST"):
            values["gateway_host"] = host
        if port := os.environ.get("WSL_GATEWAY_PORT"):
            values["gateway_port"] = int(port)
        if api_key := os.environ.get("OPENAI_API_KEY"):
            values["openai_api_key"] = api_key
        if anth_api := os.environ.get("ANTHROPIC_API_KEY"):
            values["anthropic_api_key"] = anth_api
        if gem_api := os.environ.get("GEMINI_API_KEY"):
            values["gemini_api_key"] = gem_api
        if traces := os.environ.get("TRACE_LOG_PATH"):
            values.setdefault("observability", {}).update({"trace_log_path": Path(traces)})
        if sqlite_path := os.environ.get("TRACE_SQLITE_PATH"):
            values.setdefault("observability", {}).update({"sqlite_path": Path(sqlite_path)})
        if metrics_port := os.environ.get("METRICS_PORT"):
            values.setdefault("observability", {}).update({"metrics_port": int(metrics_port)})
        if working_directory := os.environ.get("CODING_WORKING_DIRECTORY"):
            values.setdefault("coding_worker", {}).update({"working_directory": Path(working_directory)})
        if headless := os.environ.get("BROWSER_HEADLESS"):
            values.setdefault("browser_worker", {}).update({"headless": headless.lower() != "false"})
        if concurrency := os.environ.get("CODING_WORKER_CONCURRENCY"):
            values.setdefault("coding_worker", {}).update({"max_concurrency": int(concurrency)})
        if concurrency_browser := os.environ.get("BROWSER_WORKER_CONCURRENCY"):
            values.setdefault("browser_worker", {}).update({"max_concurrency": int(concurrency_browser)})
        return cls.model_validate(values)


__all__ = [
    "WindowsEdgeSettings",
    "QueueSettings",
    "ObservabilitySettings",
    "WorkerSettings",
    "CodingWorkerSettings",
    "BrowserWorkerSettings",
    "OrchestratorSettings",
]
