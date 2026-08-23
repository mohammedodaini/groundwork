.PHONY: help install run test lint typecheck eval eval-fake docker clean

# Prefer the project venv over whatever `python` happens to be on PATH.
# Without this, a system/conda interpreter runs instead and every target
# fails with ModuleNotFoundError: No module named 'groundwork'.
PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev extras
	$(PY) -m pip install -e ".[dev]"

run:  ## Run the API on :8000 (offline fake providers by default)
	$(PY) -m uvicorn groundwork.main:app --reload --port 8000

test:  ## Run the full test suite (no network, no API keys)
	$(PY) -m pytest -q

test-cov:  ## Run tests with a coverage report
	$(PY) -m pytest --cov=groundwork --cov-report=term-missing -q

lint:  ## Ruff lint + format check
	$(PY) -m ruff check src tests && $(PY) -m ruff format --check src tests

typecheck:  ## mypy
	$(PY) -m mypy src

eval:  ## Run the benchmark against the CONFIGURED provider (costs money)
	$(PY) -m groundwork.evaluation.run_eval --out docs/eval-results.md

eval-fake:  ## Run the benchmark fully offline (no network, no keys, no cost)
	$(PY) -m groundwork.evaluation.run_eval --provider fake --out /tmp/eval-smoke.md

docker:  ## Build and run with Postgres
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ groundwork.db
