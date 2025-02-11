.PHONY: setup test lint format clean help

help:  ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

setup:  ## Setup development environment
	@echo "Setting up development environment..."
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	cp .env.example .env

test:  ## Run tests
	pytest tests/ -v

lint:  ## Run code linting
	flake8 .
	mypy .
	black . --check
	isort . --check-only

format:  ## Format code
	black .
	isort .

clean:  ## Clean up temporary files and artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	rm -rf .coverage coverage.xml htmlcov/

run-api:  ## Run the API server
	uvicorn api.service:app --host $(API_HOST) --port $(API_PORT) --workers $(API_WORKERS)

download-data:  ## Download and prepare datasets
	python scripts/prepare_data.py --dataset all

train:  ## Train the model
	python scripts/run_experiments.py

# Development workflow targets
dev-setup: setup  ## Setup development environment with all tools
	pre-commit install

dev-update: ## Update development dependencies
	pip install -r requirements.txt --upgrade
	pre-commit autoupdate 