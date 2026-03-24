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
