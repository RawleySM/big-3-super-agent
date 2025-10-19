from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncGenerator

from dotenv import load_dotenv
import structlog

from .audio import AudioFrame, VADAudioStream
from .logging_utils import configure_logging, get_trace_logger
from .realtime_client import RealtimeClient, forward_transcript_to_gateway

logger = structlog.get_logger(__name__)


async def pcm_frame_generator(stream: VADAudioStream) -> AsyncGenerator[AudioFrame, None]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=16)

    def worker() -> None:
        try:
            for frame in stream.frames():
                try:
                    queue.put_nowait(frame)
                except asyncio.QueueFull:
                    logger.warning("audio.queue_full")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, AudioFrame(pcm=b"", timestamp=0.0, is_speech=False))

    loop.run_in_executor(None, worker)

    while True:
        frame = await queue.get()
        if not frame.pcm:
            break
        yield frame


async def run_loop() -> None:
    load_dotenv(".env.local")
    configure_logging(os.getenv("EDGE_LOG_LEVEL", "INFO"))
    trace_logger = get_trace_logger(os.getenv("TRACE_LOG_PATH"))

    api_key = os.environ["OPENAI_API_KEY"]
    model = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
    gateway_url = os.getenv("WSL_GATEWAY_URL", "http://127.0.0.1:8080")

    async def on_delta(event: dict) -> None:
        await forward_transcript_to_gateway(event, gateway_url, trace_logger)

    device_raw = os.getenv("AUDIO_DEVICE_INDEX")
    try:
        device = int(device_raw) if device_raw not in (None, "") else None
    except ValueError:
        logger.warning("edge.invalid_device", value=device_raw)
        device = None

    while True:
        try:
            async with RealtimeClient(api_key, model, on_delta) as client:
                with VADAudioStream(device_index=device) as stream:
                    speech_active = False
                    async for frame in pcm_frame_generator(stream):
                        if frame.is_speech:
                            if not speech_active:
                                speech_active = True
                                await client.ensure_response_started()
                                trace_logger.info(json.dumps({"event": "vad.start", "timestamp": frame.timestamp}))
                            await client.append_audio_frame(frame.pcm)
                        elif speech_active:
                            await client.commit_audio()
                            trace_logger.info(json.dumps({"event": "vad.stop", "timestamp": frame.timestamp}))
                            speech_active = False
        except Exception as exc:
            logger.error("edge.loop_error", error=str(exc))
            await asyncio.sleep(1.0)


def run() -> None:
    asyncio.run(run_loop())


if __name__ == "__main__":
    run()
