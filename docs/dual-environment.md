# Dual-Environment Orchestration Guide

This document describes how to run the Big-3 Super Agent across Windows (edge) and WSL (orchestration)
for low-latency voice control with asynchronous worker fan-out.

## 1. Windows Edge Setup

1. Install Python 3.11+ on Windows.
2. Copy `.env.local` from the repository root and fill in:
   - `OPENAI_API_KEY`
   - `REALTIME_MODEL` (defaults to `gpt-4o-realtime-preview`)
   - `WSL_GATEWAY_URL` (usually `http://127.0.0.1:8080`)
   - `TRACE_LOG_PATH` for local trace files
3. Install dependencies:
   ```powershell
   cd apps\orchestration\windows_edge
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
4. Run the edge client:
   ```powershell
   python -m windows_edge.main
   ```

The client captures microphone input, performs WebRTC-style VAD, and streams PCM frames to the
OpenAI Realtime API. Transcript deltas are forwarded immediately to WSL over HTTP without blocking.

## 2. WSL Orchestration Gateway Setup

1. Install [uv](https://docs.astral.sh/uv/) and ensure Python 3.11+ is available inside WSL.
2. Copy `.env.wsl` and set:
   - `CLAUDE_API_KEY`
   - `GEMINI_API_KEY`
   - Optional: `TRACE_LOG_PATH`, `SQLITE_PATH`
3. From the repository root run:
   ```bash
   make setup
   make dev
   ```
4. The gateway exposes:
   - `POST /task` and `POST /transcript` for task ingestion
   - `GET /status/{job_id}` for job state polling
   - `GET /metrics` for Prometheus scrapes

Workers spin up automatically and log JSON traces to the configured path.

## 3. Example Voice-to-Task Flow

1. User says: *"Generate unit tests for the new queue abstraction."*
2. Windows edge starts a `response.create` event immediately and streams audio frames.
3. The realtime API emits transcript deltas; the edge forwards them to WSL.
4. WSL gateway converts the utterance into a `coding` task and enqueues it.
5. The coding worker consumes the job, invokes Claude (placeholder in repo), and stores results.
6. `/status/{job_id}` returns `{ "status": "completed", "result": {...} }`.
7. The realtime response continues with the model's spoken reply while workers finish asynchronously.

## 4. Observability

- **Trace Log**: JSONL entries in `TRACE_LOG_PATH` with `trace_id`, `job_id`, latency, token usage, and cost estimates.
- **SQLite**: Run `make traces` to load the log into `orchestration_traces.db` for Datasette browsing.
- **Metrics**: Scrape `/metrics` for queue depth, worker completions, and error counters.

## 5. Extending the System

- Add new skills by editing `apps/orchestration/wsl_gateway/src/big3_orchestration/skills/registry.json`.
- Implement additional workers by subclassing `BaseWorker` and registering them during FastAPI startup.
- Replace placeholder Claude/Gemini calls with production SDK integrations using the provided hooks.

