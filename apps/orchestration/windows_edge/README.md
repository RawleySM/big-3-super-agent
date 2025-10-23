# Big-3 Super Agent Windows Edge

This client captures microphone audio with WebRTC-grade voice activity detection,
streams frames to the OpenAI Realtime API, and forwards transcript deltas to the WSL
orchestration gateway without blocking the user-facing conversation loop.

## Key Responsibilities

- 16 kHz PCM capture using PortAudio-compatible devices.
- Aggressive voice activity detection with configurable attack/release windows.
- Structured logging with latency and error metrics.
- Resilient streaming loop that survives network hiccups.
- Localhost HTTP/WebSocket bridge to the WSL gateway for downstream orchestration.

## Running locally

Install dependencies and start the edge loop:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m windows_edge.main
```

Environment variables are loaded from `.env.local` at the repository root.
