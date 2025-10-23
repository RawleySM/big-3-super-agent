.PHONY: setup dev lint test traces

PROJECT=apps/orchestration/wsl_gateway

venv := $(PROJECT)/.venv
UV ?= uv

setup:
	$(UV) sync --project $(PROJECT)

.dev-env: setup
	@touch .dev-env

dev: .dev-env
	$(UV) run --project $(PROJECT) --env-file .env.wsl uvicorn big3_orchestration.api.app:app --reload --host 0.0.0.0 --port 8080

lint: .dev-env
	$(UV) run --project $(PROJECT) ruff check $(PROJECT)/src $(PROJECT)/tests

test: .dev-env
	$(UV) run --project $(PROJECT) pytest $(PROJECT)/tests

TRACE_FILE ?= /tmp/big3_traces.jsonl

traces: .dev-env
	$(UV) run --project $(PROJECT) sqlite-utils insert orchestration_traces.db traces $(TRACE_FILE) --nl
