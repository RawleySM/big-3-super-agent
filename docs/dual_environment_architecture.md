# Dual-Environment Orchestration Architecture

```mermaid
flowchart TD
    Mic[Microphone Capture (Windows)] -->|20ms PCM Frames| Edge[Windows Edge Runtime]
    Edge -->|Realtime API| OpenAI[OpenAI Realtime Model]
    Edge -->|HTTP/WebSocket| Gateway[WSL FastAPI Gateway]
    Gateway -->|Queue| Queue[(Pluggable Job Queue)]
    Queue -->|Fan-out| Coding[Claude Coding Worker]
    Queue -->|Fan-out| Browser[Browser Automation Worker]
    Coding --> Traces[Unified Trace Log]
    Browser --> Traces
    Gateway -->|Metrics| Prometheus[(Metrics Endpoint)]
    Gateway -->|Status| Datasette[(SQLite / Datasette)]
    Traces -->|Fan-in| Results[Aggregated Results to Edge]
```

## Layer Responsibilities

### Windows Edge (Real-Time Loop)
- Capture microphone audio at 16 kHz mono and chunk into 20–30 ms frames.
- Apply optional WebRTC VAD to suppress silence before transmission.
- Stream audio frames to OpenAI Realtime API over WebSocket/WebRTC.
- Parse delta transcripts from the realtime stream and forward them to the WSL gateway over localhost HTTP without blocking the audio loop.
- Maintain structured JSON logging with timestamps, latencies, and VAD state transitions.
- Automatically retry the realtime connection on transient network failures.

### WSL Gateway (Agent Orchestration)
- FastAPI application (`apps/orchestrator`) exposing:
  - `POST /task` – accepts natural language instructions and enqueues tasks.
  - `GET /status/{job_id}` – return job progress and results.
  - `GET /metrics` – Prometheus-formatted counters and gauges.
  - `GET /traces` – lightweight JSON trace tailing endpoint.
- Asynchronous job queue abstraction with in-memory backend today and Redis-ready contract for the future.
- Execution engine that consumes jobs, routes them via a worker registry, and records detailed traces.
- Coding worker integration point for Claude Code or Anthropic Claude Agent SDK with sandbox-friendly stubs.
- Browser worker integration point for Gemini Computer Use or Playwright automation.
- Unified trace writer persisting JSONL logs and optional SQLite rows for Datasette exploration.

### Skill Registry & Fan-In/Fan-Out
- Natural language router converts transcripts into structured `TaskSpec` objects using extensible intent detection patterns.
- Worker registry maps each `TaskIntent` to a worker instance; new skills are added by registering an additional worker.
- Jobs can fan out by enqueuing multiple `TaskSpec` instances per utterance; results are persisted and available for fan-in aggregation through the trace log or downstream summarisation tasks.

## Observability Stack
- Structured logging via `structlog` with JSON rendering across both environments.
- `logs/windows-edge-traces.jsonl` and `logs/orchestrator-traces.jsonl` provide append-only traces.
- Optional SQLite persistence (`data/traces.sqlite`) enables Datasette dashboards.
- Prometheus-friendly metrics served at `/metrics` cover queue depth, completions, and failures.

## Example Voice-to-Task Flow
1. User speaks: “Refactor the orchestrator router and open the QA dashboard to verify deploy status.”
2. Windows edge streams audio to OpenAI Realtime, receives transcript deltas, and immediately posts them to `/task` on WSL.
3. Router recognises coding + browser intents, enqueues separate jobs, and the execution engine fans out to the respective workers.
4. Coding worker delegates to Claude Code to prepare a patch proposal; browser worker launches Gemini Computer Use automation to validate the dashboard.
5. Both workers write JSONL traces with job metadata, execution timings, token usage, and cost estimates.
6. Gateway collects results and exposes them via `/status/{job_id}` and `/traces`; Datasette and dashboards can visualise trends.

Refer to the repository README for setup instructions, environment configuration, and operational commands.
