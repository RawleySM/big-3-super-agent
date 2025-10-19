from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import structlog


def configure_logging(level: str | int = "INFO") -> None:
    """Configure structlog for JSON logs with UTC timestamps."""

    if isinstance(level, str):
        level = level.upper()
    logging.basicConfig(level=level, format="%(message)s")

    def add_timestamp(_: Any, __: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        return event_dict

    structlog.configure(
        processors=[
            add_timestamp,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )


def get_trace_logger(trace_path: str | None) -> logging.Logger:
    """Return a standard logger that appends JSON lines to the trace file."""
    logger = logging.getLogger("trace")
    if trace_path:
        path = Path(os.path.expandvars(trace_path)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
