"""Prometheus-style metrics collector."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Dict


class MetricsRegistry:
    """In-memory metrics registry."""

    def __init__(self) -> None:
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._lock = asyncio.Lock()

    async def incr(self, key: str, amount: float = 1.0) -> None:
        async with self._lock:
            self._counters[key] += amount

    async def set_gauge(self, key: str, value: float) -> None:
        async with self._lock:
            self._gauges[key] = value

    async def export(self) -> str:
        async with self._lock:
            lines = []
            for key, value in self._counters.items():
                lines.append(f"# TYPE {key} counter")
                lines.append(f"{key} {value}")
            for key, value in self._gauges.items():
                lines.append(f"# TYPE {key} gauge")
                lines.append(f"{key} {value}")
            return "\n".join(lines) + "\n"


_metrics_registry: MetricsRegistry | None = None


def get_metrics_registry() -> MetricsRegistry:
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()
    return _metrics_registry
