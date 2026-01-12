.PHONY: help install dev format lint clean jupyter

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install dependencies using uv
	uv sync

dev:  ## Install development dependencies
	uv sync --all-extras

format:  ## Format code with ruff
	uv run ruff format .
	uv run nbqa ruff tutorials/ --fix

lint:  ## Lint code with ruff
	uv run ruff check .
	uv run nbqa ruff tutorials/

clean:  ## Clean cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

jupyter:  ## Launch Jupyter notebook
	uv run jupyter notebook
