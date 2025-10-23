from __future__ import annotations

import asyncio
import base64
import json
from typing import Awaitable, Callable, Optional

import httpx
import structlog
import websockets
from websockets import WebSocketClientProtocol

logger = structlog.get_logger(__name__)

RealtimeEventHandler = Callable[[dict], Awaitable[None]]


class RealtimeClient:
    """Minimal client for the OpenAI Realtime WebSocket API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        on_transcript_delta: RealtimeEventHandler,
        session_instructions: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._on_transcript_delta = on_transcript_delta
        self._session_instructions = session_instructions
        self._ws: Optional[WebSocketClientProtocol] = None
        self._response_open = False
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "RealtimeClient":
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        self._ws = await websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={self._model}",
            extra_headers=headers,
            subprotocols=["realtime"],
            open_timeout=5,
            ping_interval=10,
            ping_timeout=10,
            max_queue=32,
        )
        await self._ws.send(json.dumps({"type": "session.update", "session": {"modalities": ["text", "audio"], "instructions": self._session_instructions or ""}}))
        asyncio.create_task(self._receive_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._ws:
            await self._ws.close()
        self._ws = None
        self._response_open = False

    async def _receive_loop(self) -> None:
        assert self._ws
        try:
            async for raw in self._ws:
                event = json.loads(raw)
                if event.get("type") == "response.delta":
                    await self._on_transcript_delta(event)
                elif event.get("type") == "response.completed":
                    self._response_open = False
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("realtime.receive_error", error=str(exc))
            self._response_open = False

    async def ensure_response_started(self) -> None:
        async with self._lock:
            if self._response_open:
                return
            await self._send({"type": "response.create", "response": {"modalities": ["text", "audio"], "conversation": "default"}})
            self._response_open = True
            logger.info("realtime.response_started")

    async def append_audio_frame(self, pcm_bytes: bytes) -> None:
        encoded = base64.b64encode(pcm_bytes).decode("ascii")
        await self._send({"type": "input_audio_buffer.append", "audio": encoded})

    async def commit_audio(self) -> None:
        await self._send({"type": "input_audio_buffer.commit"})

    async def _send(self, payload: dict) -> None:
        if not self._ws:
            raise RuntimeError("Realtime client not connected")
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as exc:
            logger.error("realtime.send_failed", error=str(exc))
            raise


async def forward_transcript_to_gateway(event: dict, gateway_url: str, trace_logger) -> None:
    if event.get("type") != "response.delta":
        return
    outputs = event.get("delta", {}).get("output", [])
    fragments = []
    for item in outputs:
        if item.get("type") == "output_text.delta":
            fragments.append(item.get("text", ""))
        elif item.get("type") == "input_text.delta":
            fragments.append(item.get("text", ""))
    if not fragments:
        return
    transcript = "".join(fragments)
    payload = {"transcript": transcript, "event": event}
    async with httpx.AsyncClient(timeout=1.0) as client:
        try:
            response = await client.post(f"{gateway_url}/transcript", json=payload)
            response.raise_for_status()
        except Exception as exc:
            trace_logger.warning(json.dumps({"event": "transcript.forward_failed", "error": str(exc)}))
            logger.warning("gateway.forward_failed", error=str(exc))
        else:
            trace_logger.info(json.dumps({"event": "transcript.forwarded", "length": len(transcript)}))
            logger.debug("gateway.forwarded", chars=len(transcript))
