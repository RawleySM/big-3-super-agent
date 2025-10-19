.PHONY: dev lint test traces

UV ?= uv

ROOT_DIR := $(shell pwd)
ORCH_DIR := $(ROOT_DIR)/apps/orchestration

export PYTHONPATH := $(ORCH_DIR):$(PYTHONPATH)

dev:
cd $(ORCH_DIR) && $(UV) run uvicorn dual_orchestrator.main:app --reload --host 0.0.0.0 --port 8080

lint:
cd $(ORCH_DIR) && $(UV) run ruff check dual_orchestrator

lint-fix:
cd $(ORCH_DIR) && $(UV) run ruff check --fix dual_orchestrator

test:
cd $(ORCH_DIR) && $(UV) run pytest

traces:
cd $(ORCH_DIR) && $(UV) run python -m dual_orchestrator.utils.traces_to_sqlite logs/traces.jsonl logs/traces.db
