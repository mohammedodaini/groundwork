.PHONY: help install run test lint typecheck eval eval-fake docker clean lock lock-check smoke audit

# Prefer the project venv over whatever `python` happens to be on PATH.
# Without this, a system/conda interpreter runs instead and every target
# fails with ModuleNotFoundError: No module named 'groundwork'.
PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev extras, pinned by requirements-dev.lock
	$(PY) -m pip install -c requirements-dev.lock -e ".[dev]"

lock:  ## Re-resolve the lockfiles after changing dependencies in pyproject.toml
	uv pip compile pyproject.toml --universal --python-version 3.11 --extra anthropic --extra openai --extra postgres -o requirements.lock
	uv pip compile pyproject.toml --universal --python-version 3.11 --extra dev -o requirements-dev.lock

audit:  ## Check the locked dependencies against known vulnerabilities (same check as CI)
	uvx pip-audit==2.10.1 -r requirements.lock -r requirements-dev.lock --disable-pip --no-deps --progress-spinner off

lock-check:  ## Fail if the lockfiles are stale (existing pins are kept; only pyproject changes move them)
	$(MAKE) lock
	git diff --exit-code requirements.lock requirements-dev.lock

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

smoke:  ## End-to-end check against a running instance: make smoke URL=http://localhost:8000
	$(PY) scripts/smoke_test.py $(or $(URL),http://localhost:8000) $(if $(API_KEY),--api-key $(API_KEY))
