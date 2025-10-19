from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import aiohttp
import sounddevice as sd
import webrtcvad
from aiohttp import ClientSession, WSMsgType

from ..config import WindowsEdgeSettings, _load_env_file

logger = logging.getLogger("dual_orchestrator.windows_edge")


@dataclass(slots=True)
class PartialTranscript:
    text: str
    is_final: bool
    latency_ms: float


class MicrophoneStream:
    def __init__(self, sample_rate: int, frame_ms: int) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self._stream: Optional[sd.InputStream] = None

    async def __aenter__(self) -> "MicrophoneStream":
        loop = asyncio.get_event_loop()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
        )
        await loop.run_in_executor(None, self._stream.start)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stream:
            await asyncio.get_event_loop().run_in_executor(None, self._stream.stop)
            self._stream.close()

    async def frames(self) -> AsyncIterator[bytes]:
        if not self._stream:
            raise RuntimeError("Stream not started")
        loop = asyncio.get_event_loop()
        while True:
            data, _ = await loop.run_in_executor(None, self._stream.read, self.frame_samples)
            yield data.tobytes()


class VoiceEdgeClient:
    def __init__(self, settings: WindowsEdgeSettings) -> None:
        self.settings = settings
        self.vad = webrtcvad.Vad(3)
        self._session: Optional[ClientSession] = None
        self._last_speech_time = 0.0

    async def __aenter__(self) -> "VoiceEdgeClient":
        self._session = aiohttp.ClientSession(headers={"Authorization": f"Bearer {self._api_key}"})
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    @property
    def _api_key(self) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for realtime streaming")
        return api_key

    async def _stream_to_realtime(self, frames: AsyncIterator[bytes]) -> AsyncIterator[PartialTranscript]:
        assert self._session is not None
        params = {"model": self.settings.realtime_model}
        async with self._session.ws_connect(self.settings.realtime_api_url, params=params) as ws:
            await ws.send_json({"type": "response.create"})
            async for frame in frames:
                is_speech = self.vad.is_speech(frame, self.settings.microphone_sample_rate)
                await ws.send_bytes(frame)
                if is_speech:
                    self._last_speech_time = time.time()
                while True:
                    try:
                        msg = await ws.receive(timeout=0.01)
                    except asyncio.TimeoutError:
                        break
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    event_type = data.get("type")
                    if event_type == "response.delta":
                        text = data.get("delta", {}).get("text", "")
                        if text:
                            latency = (time.time() - self._last_speech_time) * 1000
                            yield PartialTranscript(text=text, is_final=False, latency_ms=latency)
                    elif event_type == "response.completed":
                        latency = (time.time() - self._last_speech_time) * 1000
                        yield PartialTranscript(text=data.get("response", ""), is_final=True, latency_ms=latency)
                        return

    async def stream(self) -> None:
        _load_env_file(".env.local")
        while True:
            try:
                async with self as client:
                    async with MicrophoneStream(self.settings.microphone_sample_rate, self.settings.frame_ms) as mic:
                        async for transcript in self._stream_to_realtime(mic.frames()):
                            await self._send_to_gateway(transcript)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Realtime streaming interrupted, retrying", exc_info=exc)
                await asyncio.sleep(self.settings.reconnect_backoff_seconds)

    async def _send_to_gateway(self, transcript: PartialTranscript) -> None:
        payload = {
            "text": transcript.text,
            "is_final": transcript.is_final,
            "latency_ms": transcript.latency_ms,
            "timestamp": time.time(),
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.settings.local_gateway_url}/transcript", json=payload, timeout=1) as resp:
                    if resp.status >= 400:
                        logger.warning("Gateway responded with %s", resp.status)
            except asyncio.TimeoutError:
                logger.warning("Gateway timeout when forwarding transcript", extra={"payload": payload})
            except Exception:  # pragma: no cover - defensive
                logger.exception("Unexpected error when forwarding transcript")


async def main() -> None:  # pragma: no cover - interactive script
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    settings = WindowsEdgeSettings.from_env()
    client = VoiceEdgeClient(settings)
    await client.stream()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
