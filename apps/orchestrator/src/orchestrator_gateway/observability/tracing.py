"""Tracing utilities for structured logging."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

import aiosqlite
import structlog

from ..config import get_settings
from ..models import TraceRecord

_logger = structlog.get_logger(__name__)


@dataclass
class TraceSpan:
    trace_id: str
    job_id: str
    component: str
    payload: Dict
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_record(self) -> TraceRecord:
        return TraceRecord(
            timestamp=self.timestamp,
            trace_id=self.trace_id,
            job_id=self.job_id,
            component=self.component,  # type: ignore[arg-type]
            payload=self.payload,
        )


class TraceWriter:
    """Writes trace records to JSONL and optionally SQLite."""

    def __init__(self, log_path: Path, db_path: Path | None = None) -> None:
        self.log_path = log_path
        self.db_path = db_path
        self._lock = asyncio.Lock()

    async def write(self, span: TraceSpan) -> None:
        record = span.to_record()
        async with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        if self.db_path:
            await self._write_sqlite(record)

    async def _write_sqlite(self, record: TraceRecord) -> None:
        db_path = self.db_path
        if db_path is None:
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    timestamp TEXT,
                    trace_id TEXT,
                    job_id TEXT,
                    component TEXT,
                    payload TEXT
                )
                """
            )
            await db.execute(
                "INSERT INTO traces VALUES (?, ?, ?, ?, ?)",
                (
                    record.timestamp.isoformat(),
                    record.trace_id,
                    record.job_id,
                    record.component,
                    json.dumps(record.payload),
                ),
            )
            await db.commit()


_trace_writer: TraceWriter | None = None


def get_trace_writer() -> TraceWriter:
    global _trace_writer
    if _trace_writer is None:
        settings = get_settings()
        _trace_writer = TraceWriter(settings.trace_log_path, settings.trace_db_path)
    return _trace_writer


@asynccontextmanager
async def traced_span(trace_id: str, job_id: str, component: str, payload: Optional[Dict] = None) -> AsyncIterator[None]:
    payload = payload or {}
    start = datetime.utcnow()
    writer = get_trace_writer()
    await writer.write(TraceSpan(trace_id=trace_id, job_id=job_id, component=component, payload={"event": "start", **payload}))
    try:
        yield
        elapsed = (datetime.utcnow() - start).total_seconds()
        await writer.write(
            TraceSpan(
                trace_id=trace_id,
                job_id=job_id,
                component=component,
                payload={"event": "end", "elapsed_seconds": elapsed, **payload},
            )
        )
    except Exception as exc:  # pragma: no cover - logging path
        _logger.exception("Span failed", trace_id=trace_id, job_id=job_id, component=component, error=str(exc))
        await writer.write(
            TraceSpan(
                trace_id=trace_id,
                job_id=job_id,
                component=component,
                payload={"event": "error", "error": str(exc), **payload},
            )
        )
        raise
