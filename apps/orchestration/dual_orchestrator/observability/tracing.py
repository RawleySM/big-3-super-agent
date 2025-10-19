from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from prometheus_client import Counter, Histogram

TRACE_WRITE_LATENCY = Histogram(
    "dual_orchestrator_trace_write_seconds", "Time spent writing trace events"
)
TRACE_EVENTS = Counter(
    "dual_orchestrator_trace_events_total", "Number of trace events written", ["phase"]
)


class TraceLogger:
    """Append-only JSONL trace logger with Prometheus metrics."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, trace_id: str, job_id: str, *, phase: str, payload: Optional[Dict[str, Any]] = None) -> None:
        payload = payload or {}
        record = {
            "timestamp": time.time(),
            "trace_id": trace_id,
            "job_id": job_id,
            "phase": phase,
            "payload": payload,
        }
        with TRACE_WRITE_LATENCY.time():
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, default=str) + "\n")
        TRACE_EVENTS.labels(phase=phase).inc()


__all__ = ["TraceLogger", "TRACE_EVENTS", "TRACE_WRITE_LATENCY"]
