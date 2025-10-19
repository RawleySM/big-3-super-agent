# Big-3 Super Agent Orchestration Gateway (WSL)

This service runs inside WSL and provides the HTTP/WebSocket gateway that receives
structured tasks from the Windows edge process, fans them out to the appropriate
long-running workers, and aggregates their results.

## Features

- FastAPI application with `/task`, `/status/{job_id}`, and `/metrics` endpoints.
- Pluggable job queue abstraction (in-memory by default, Redis ready).
- Dedicated worker loops for the coding and browser agents with sandbox hooks for
  invoking Anthropic Claude and Gemini Computer Use APIs.
- Unified JSONL trace logging with latency, token usage, and cost tracking metadata.
- Optional SQLite persistence of traces for later analysis with Datasette.
- Prometheus-compatible metrics for queue depth, worker throughput, and error rates.

## Development

```bash
uv sync --project apps/orchestration/wsl_gateway
uv run --project apps/orchestration/wsl_gateway --env-file ../../..//.env.wsl uvicorn big3_orchestration.api.app:app --reload
```

Or use the root `Makefile`:

```bash
make dev
make lint
make test
```

## Trace inspection

To materialise the unified trace log into SQLite, run:

```bash
make traces
```

and then open Datasette:

```bash
uv run --project apps/orchestration/wsl_gateway datasette orchestration_traces.db
```

