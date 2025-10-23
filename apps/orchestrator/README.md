# Big-3 Orchestrator

The orchestrator package hosts the WSL-only gateway that receives natural language tasks from the Windows edge layer, routes those tasks to specialized workers (coding and browser automation), tracks execution and costs, and exposes observability endpoints.

Key features:

- FastAPI gateway with `/task`, `/status/{job_id}`, `/metrics`, and `/traces` endpoints.
- Pluggable asynchronous job queue abstraction, with an in-memory implementation shipping out of the box and Redis-friendly contracts for future use.
- Modular worker registry that maps task intents to agent workers (coding, browser automation, summarisation, etc.).
- JSONL trace logging and optional SQLite persistence for Datasette-friendly analysis.
- Structured logging via `structlog` and consistent tracing metadata (`trace_id`, `job_id`, timing, token usage, cost).

Refer to the repository root documentation for setup and architecture diagrams.
