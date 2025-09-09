.PHONY: help install dev test lint format type-check security clean build run docker-build docker-run setup

# Default target
help:
	@echo "AURA - Autonomous Unified Research Assistant"
	@echo ""
	@echo "Available targets:"
	@echo "  setup          Setup development environment"
	@echo "  install        Install dependencies"
	@echo "  dev            Install development dependencies"
	@echo "  test           Run tests"
	@echo "  lint           Run linting"
	@echo "  format         Format code"
	@echo "  type-check     Run type checking"
	@echo "  security       Run security checks"
	@echo "  clean          Clean build artifacts"
	@echo "  build          Build package"
	@echo "  run            Run AURA"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run with Docker Compose"

setup:
	@chmod +x scripts/setup.sh
	@./scripts/setup.sh

install:
	uv sync

dev:
	uv sync --dev

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=aura --cov-report=html --cov-report=term

lint:
	uv run ruff check .

format:
	uv run ruff format .

type-check:
	uv run mypy aura/

security:
	uv run bandit -r aura/
	uv run safety check

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

build:
	uv build

run:
	uv run python -m aura.main

docker-build:
	docker build -t aura:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f aura