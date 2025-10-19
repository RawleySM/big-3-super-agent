PYTHON ?= python3
POETRY ?= poetry
ORCHESTRATOR_DIR := apps/orchestrator
WINDOWS_EDGE_DIR := apps/windows-edge

.PHONY: setup dev lint test traces

setup:
cd $(ORCHESTRATOR_DIR) && $(POETRY) install

dev:
cd $(ORCHESTRATOR_DIR) && $(POETRY) run uvicorn orchestrator_gateway.main:app --reload --host $${FASTAPI_HOST:-0.0.0.0} --port $${FASTAPI_PORT:-8000}

lint:
cd $(ORCHESTRATOR_DIR) && $(POETRY) run ruff check src


test:
cd $(ORCHESTRATOR_DIR) && $(POETRY) run pytest

traces:
@echo "Trace log located at: $${TRACE_LOG_PATH:-logs/orchestrator-traces.jsonl}"
