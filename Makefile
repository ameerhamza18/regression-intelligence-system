.PHONY: setup lint format test

setup:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .

format:
	ruff format .

test:
	pytest --cov=regression_intelligence


validate:
	python scripts/run_validation.py
