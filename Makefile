.PHONY: format lint test install dev clean

format:
	black src tests
	isort src tests

lint:
	ruff check src tests
	mypy src

test:
	pytest tests -v

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

clean:
	rm -rf build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.PHONY: ci ci-docker

ci:
	uv run black --check src tests
	uv run isort --check-only src tests
	uv run ruff check src tests
	uv run mypy src
	uv run pytest tests -v
	uv build

ci-docker:
	set -eu; for version in 3.12 3.14; do \
		docker run --rm -v "$(CURDIR):/work" -w /work \
			-e UV_PROJECT_ENVIRONMENT=/tmp/pyppann-venv \
			-e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
			python:$$version sh -c \
			"pip install uv && uv sync --locked --extra dev && make ci"; \
	done
