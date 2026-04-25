# Run from repository root. Requires GNU Make and Python 3.10+.
PYTHON ?= python
PIP ?= $(PYTHON) -m pip

# Make both local DQN package and env-setup shared src importable in make targets.
export PYTHONPATH := $(CURDIR)/dqn/src;$(CURDIR)

.PHONY: help install install-dev test test-env lint format format-check train eval eval-watch eval-gif eval-viz clean check ci dev all

# Optional path for eval-gif / eval-viz (override: make eval-gif GIF=artifacts/run.gif)
GIF ?= results/dqn/eval_episode.gif

help:
	@echo "Layout: DQN under dqn/; shared env setup from env-setup under src/ and tests/."
	@echo "Targets:"
	@echo "  install      - pip install -r requirements.txt and editable package (dqn/src/)"
	@echo "  install-dev  - same as install"
	@echo "  test         - all pytest tests (dqn/tests + tests)"
	@echo "  test-env     - env-setup sanity test only"
	@echo "  lint         - ruff check"
	@echo "  format       - ruff format"
	@echo "  format-check - ruff format --check"
	@echo "  train        - run DQN training (python -m DQN.main)"
	@echo "  eval         - greedy eval only (override: make eval ARGS='--episodes 50')"
	@echo "  eval-watch   - eval then one live greedy episode (pygame; ARGS passed through)"
	@echo "  eval-gif     - eval then save one greedy episode to GIF=path (default: results/dqn/eval_episode.gif)"
	@echo "  eval-viz     - eval then GIF + live; ARGS e.g. --demo-trials 64 or --demo-trials 0 --vis-seed 42"
	@echo "  clean        - remove caches and build artifacts"
	@echo "  check / ci   - lint + tests (CI-style)"
	@echo "  dev          - format then check (local pre-push)"
	@echo "  all          - format, lint, test"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: install

test:
	$(PYTHON) -m pytest

test-env:
	$(PYTHON) -m pytest tests/test_env.py

lint:
	$(PYTHON) -m ruff check dqn/src dqn/tests src tests

format:
	$(PYTHON) -m ruff format dqn/src dqn/tests src tests

format-check:
	$(PYTHON) -m ruff format --check dqn/src dqn/tests src tests

train:
	$(PYTHON) -m DQN.main

eval:
	$(PYTHON) -m DQN.eval $(ARGS)

eval-watch:
	$(PYTHON) -m DQN.eval --watch $(ARGS)

eval-gif:
	$(PYTHON) -m DQN.eval --gif "$(GIF)" $(ARGS)

eval-viz:
	$(PYTHON) -m DQN.eval --gif "$(GIF)" --watch $(ARGS)

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.ruff_cache')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('*.egg-info')]"
	@echo Cleaned Python caches.

check: lint test

ci: check

dev: format check

all: format lint test
