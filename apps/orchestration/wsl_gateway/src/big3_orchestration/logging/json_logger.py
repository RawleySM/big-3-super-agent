from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import structlog


def configure_logging(trace_path: Path | None) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

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

    trace_logger = logging.getLogger("trace")
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(trace_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        trace_logger.addHandler(handler)
    trace_logger.setLevel(logging.INFO)
    return trace_logger


def emit_trace(trace_logger: logging.Logger, **payload: Any) -> None:
    trace_logger.info(json.dumps(payload))
