"""Windows edge process for realtime audio streaming."""
from __future__ import annotations

import asyncio
import json
import os
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
from dotenv import load_dotenv
import aiohttp

try:  # pragma: no cover - platform specific
    import sounddevice as sd
except ImportError:  # pragma: no cover - platform specific
    sd = None

try:  # pragma: no cover - platform specific
    import webrtcvad
except ImportError:  # pragma: no cover - platform specific
    webrtcvad = None

try:
    import websockets
except ImportError:  # pragma: no cover - fallback to aiohttp
    websockets = None


LOGGER = structlog.get_logger(__name__)


@dataclass
class EdgeSettings:
    openai_api_key: str
    openai_model: str = "gpt-4o-realtime-preview"
    stream_host: str = "127.0.0.1"
    stream_port: int = 9099
    gateway_url: str = "http://127.0.0.1:8000"
    vad_aggressiveness: int = 2
    frame_ms: int = 20
    sample_rate: int = 16_000
    channels: int = 1
    trace_log_path: Path = Path("logs/windows-edge-traces.jsonl")

    @classmethod
    def from_env(cls) -> "EdgeSettings":
        load_dotenv(Path(__file__).resolve().parents[2] / ".env.local")
        return cls(
            openai_api_key=os.environ.get("OPENAI_REALTIME_API_KEY", ""),
            openai_model=os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview"),
            stream_host=os.environ.get("EDGE_STREAM_HOST", "127.0.0.1"),
            stream_port=int(os.environ.get("EDGE_STREAM_PORT", "9099")),
            gateway_url=os.environ.get("WSL_GATEWAY_URL", "http://127.0.0.1:8000"),
            vad_aggressiveness=int(os.environ.get("VAD_AGGRESSIVENESS", "2")),
            trace_log_path=Path(os.environ.get("TRACE_LOG_PATH", "logs/windows-edge-traces.jsonl")),
        )


class VoiceStreamer:
    """Captures microphone audio, performs VAD, and streams frames."""

    def __init__(self, settings: EdgeSettings) -> None:
        self.settings = settings
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False
        self._last_vad_state = False
        if webrtcvad:
            self._vad = webrtcvad.Vad(settings.vad_aggressiveness)
        else:
            self._vad = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: Optional[sd.CallbackFlags]) -> None:
        if status:  # pragma: no cover - logging path
            LOGGER.warning("audio callback status", status=str(status))
        # sounddevice delivers float32 samples in range [-1, 1]; convert to int16 PCM
        audio = np.clip(indata.copy(), -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
        self._audio_queue.put(audio)

    async def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is required on Windows edge host")
        if self._running:
            return
        self._running = True
        self.settings.trace_log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = sd.InputStream(
            samplerate=self.settings.sample_rate,
            blocksize=int(self.settings.sample_rate * self.settings.frame_ms / 1000),
            channels=self.settings.channels,
            dtype="int16",
            callback=self._audio_callback,
        )
        stream.start()
        LOGGER.info("voice streamer started", sample_rate=self.settings.sample_rate)
        append_trace(self.settings, "voice_streamer.start", sample_rate=self.settings.sample_rate)
        try:
            while self._running:
                await asyncio.sleep(0.01)
        finally:  # pragma: no cover - cleanup path
            stream.stop()
            stream.close()
            append_trace(self.settings, "voice_streamer.stop")

    def stop(self) -> None:
        self._running = False

    def get_frame(self) -> Optional[np.ndarray]:
        try:
            return self._audio_queue.get_nowait()
        except queue.Empty:
            return None

    def is_speech(self, frame: np.ndarray) -> bool:
        if self._vad is None:
            speech = True
        else:
            bytes_frame = frame.tobytes()
            try:
                speech = self._vad.is_speech(bytes_frame, self.settings.sample_rate)
            except Exception:  # pragma: no cover - vad failure fallback
                speech = True
        if speech != self._last_vad_state:
            self._last_vad_state = speech
            LOGGER.info("vad state change", active=speech)
            append_trace(self.settings, "vad.state", active=speech)
        return speech

    @property
    def running(self) -> bool:
        return self._running


async def stream_to_openai(settings: EdgeSettings, voice_streamer: VoiceStreamer) -> None:
    """Connect to OpenAI realtime endpoint and push audio frames."""

    async def send_loop(ws) -> None:
        LOGGER.info("send loop started")
        append_trace(settings, "edge.send_loop.start")
        first_packet_sent = False
        while voice_streamer.running:
            frame = voice_streamer.get_frame()
            if frame is None:
                await asyncio.sleep(0.005)
                continue
            if not voice_streamer.is_speech(frame):
                continue
            payload = frame.tobytes()
            if not first_packet_sent:
                initial_event = json.dumps({"type": "response.create", "response": {"modalities": ["text"]}})
                if websockets:
                    await ws.send(initial_event)
                else:
                    await ws.send_str(initial_event)
                append_trace(settings, "edge.response_create")
            if websockets:
                await ws.send(payload)
            else:
                await ws.send_bytes(payload)
            if not first_packet_sent:
                LOGGER.info("first audio frame sent")
                append_trace(settings, "edge.first_frame")
                first_packet_sent = True

    async def receive_loop(ws) -> None:
        LOGGER.info("receive loop started")
        append_trace(settings, "edge.receive_loop.start")
        while voice_streamer.running:
            if websockets:
                message = await ws.recv()
            else:
                message = await ws.receive()
            LOGGER.info("openai message", message=message)
            await forward_transcript(message, settings)

    if websockets:
        url = f"wss://api.openai.com/v1/realtime?model={settings.openai_model}"
        async with websockets.connect(
            url,
            extra_headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            max_size=None,
        ) as ws:
            await asyncio.gather(send_loop(ws), receive_loop(ws))
    else:  # pragma: no cover - aiohttp fallback
        url = f"wss://api.openai.com/v1/realtime?model={settings.openai_model}"
        async with aiohttp.ClientSession(headers={"Authorization": f"Bearer {settings.openai_api_key}"}) as session:
            async with session.ws_connect(url, autoping=True) as ws:
                await asyncio.gather(send_loop(ws), receive_loop(ws))


async def forward_transcript(message: str, settings: EdgeSettings) -> None:
    """Send partial transcripts to the WSL gateway."""
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        LOGGER.debug("non json message", message=message)
        return
    transcript = data.get("transcript") or data.get("text")
    if not transcript:
        return
    async with aiohttp.ClientSession() as session:
        await session.post(
            f"{settings.gateway_url}/task",
            json={"utterance": transcript, "context": {"source": "voice"}},
            timeout=5,
        )
    append_trace(settings, "edge.forward_transcript", transcript=transcript)


async def run_edge() -> None:
    settings = EdgeSettings.from_env()
    if not settings.openai_api_key:
        LOGGER.error("OPENAI_REALTIME_API_KEY not configured")
        return
    voice_streamer = VoiceStreamer(settings)
    streamer_task = asyncio.create_task(voice_streamer.start())
    try:
        while True:
            try:
                await stream_to_openai(settings, voice_streamer)
            except Exception as exc:  # pragma: no cover - runtime path
                LOGGER.exception("streaming failed", error=str(exc))
                append_trace(settings, "edge.error", error=str(exc))
                await asyncio.sleep(2.0)
            else:
                await asyncio.sleep(1.0)
    finally:
        voice_streamer.stop()
        await streamer_task


def append_trace(settings: EdgeSettings, event: str, **payload) -> None:
    """Append a trace event to the configured JSONL log."""
    record = {"event": event, "timestamp": time.time(), **payload}
    settings.trace_log_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.trace_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":  # pragma: no cover
    structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()])
    try:
        asyncio.run(run_edge())
    except KeyboardInterrupt:
        LOGGER.info("edge client interrupted")
