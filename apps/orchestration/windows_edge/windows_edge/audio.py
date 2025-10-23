from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np
import sounddevice as sd
import webrtcvad

FRAME_DURATION_MS = 20
TARGET_SAMPLE_RATE = 16_000


def _bytes_from_frame(frame: np.ndarray) -> bytes:
    # Convert float32 samples to 16-bit PCM little endian
    ints = np.clip(frame * 32767, -32768, 32767).astype(np.int16)
    return ints.tobytes()


@dataclass
class AudioFrame:
    pcm: bytes
    timestamp: float
    is_speech: bool


class VADAudioStream:
    """Capture audio and yield VAD-tagged frames suitable for realtime APIs."""

    def __init__(
        self,
        device_index: Optional[int] = None,
        frame_ms: int = FRAME_DURATION_MS,
        aggressiveness: int = 2,
    ) -> None:
        self.frame_ms = frame_ms
        self.sample_rate = TARGET_SAMPLE_RATE
        self.block_size = int(self.sample_rate * frame_ms / 1000)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._device_index = device_index
        self._vad = webrtcvad.Vad(aggressiveness)
        self._running = threading.Event()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # type: ignore[override]
        if status:
            # Drop status info; main loop logs warnings on lag.
            pass
        self._queue.put(indata.copy())

    def __enter__(self) -> "VADAudioStream":
        self._running.set()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            channels=1,
            callback=self._callback,
            device=self._device_index,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self._running.clear()
        if self._stream:
            self._stream.stop()
            self._stream.close()
        while not self._queue.empty():
            self._queue.get_nowait()

    def frames(self) -> Generator[AudioFrame, None, None]:
        while self._running.is_set():
            try:
                chunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            pcm = _bytes_from_frame(chunk[:, 0])
            is_speech = self._vad.is_speech(pcm, self.sample_rate)
            yield AudioFrame(pcm=pcm, timestamp=time.time(), is_speech=is_speech)
