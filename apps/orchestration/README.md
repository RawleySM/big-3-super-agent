# Big-3 Dual-Orchestration App

This package contains the FastAPI orchestration gateway, worker infrastructure, and Windows voice edge client that form the dual-environment runtime for the Big-3 Super Agent.

## Quickstart (WSL)

```bash
# Install dependencies
uv sync

# Start the orchestration gateway
make dev
```

- **FastAPI Gateway:** `http://localhost:8080`
- **Metrics:** `http://localhost:8080/metrics`
- **Health Check:** `http://localhost:8080/health`

## Quickstart (Windows Edge)

```powershell
# Inside apps/orchestration
py -3 -m pip install -e .[windows]
py -3 dual_orchestrator/windows_edge/edge_client.py
```

The edge client streams microphone audio to the OpenAI Realtime API and forwards transcripts to the WSL gateway. Logs capture VAD state, latency, and reconnection events.

## Configuration

- `.env.local` – Windows-specific runtime variables (realtime key, VAD tuning, gateway URL)
- `.env.wsl` – WSL orchestration credentials and directories

All configuration files are opt-in; no secrets are committed to the repository.

## Observability & Traces

- `logs/traces.jsonl` – append-only trace log with per-job metadata
- `make traces` – convert JSONL to SQLite for Datasette browsing
- Prometheus metrics for trace volume and write latency

## Tests & Tooling

```bash
make lint
make test
```

Tests cover the task router, job store, and fan-out orchestration logic.
