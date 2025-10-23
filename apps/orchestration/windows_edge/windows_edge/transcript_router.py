from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable

import httpx
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TranscriptFragment:
    text: str
    timestamp: float


class TranscriptBuffer:
    """Buffers transcript fragments and periodically ships them to WSL."""

    def __init__(self, gateway_url: str, flush_interval: float = 0.35) -> None:
        self._gateway_url = gateway_url.rstrip("/")
        self._flush_interval = flush_interval
        self._buffer: Deque[TranscriptFragment] = deque()
        self._lock = asyncio.Lock()

    async def add_fragment(self, text: str) -> None:
        async with self._lock:
            self._buffer.append(TranscriptFragment(text=text, timestamp=time.time()))

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval)
            await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if not self._buffer:
                return
            fragments = list(self._buffer)
            self._buffer.clear()
        payload = {
            "utterance": "".join(fragment.text for fragment in fragments),
            "fragments": [fragment.__dict__ for fragment in fragments],
        }
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.post(f"{self._gateway_url}/transcript", json=payload)
                response.raise_for_status()
        except Exception as exc:
            logger.warning("transcript.flush_failed", error=str(exc))
        else:
            logger.info("transcript.flushed", chars=len(payload["utterance"]))
