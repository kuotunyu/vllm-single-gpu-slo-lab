UV ?= uv

.PHONY: help sync test lint format audit-secrets reproduce-lite reproduce

help:
	@echo "sync            uv sync --all-extras"
	@echo "test            pytest"
	@echo "lint            ruff check + ruff format --check"
	@echo "format          ruff format + ruff check --fix"
	@echo "audit-secrets   scan the tree for IPs / keys / tokens (fails on findings)"
	@echo "reproduce-lite  CPU-only: validate configs, rebuild whatever evidence exists"
	@echo "reproduce       reproduce-lite, then require a zero git diff under evidence/ and analysis/"

sync:
	$(UV) sync --all-extras

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

audit-secrets:
	$(UV) run python scripts/redact.py audit .

reproduce-lite:
	$(UV) run slo-lab reproduce-lite --root .

reproduce: reproduce-lite
	git diff --exit-code --stat -- evidence analysis
