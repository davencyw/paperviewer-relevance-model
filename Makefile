# Variables
PYTHON_FILES := src  # Adjust this to target specific directories or files

# Commands
.PHONY: lint format type-check all clean

lint:
	@echo "Running ruff for linting..."
	ruff check $(PYTHON_FILES)
	@echo "Running mypy for type checking..."
	mypy $(PYTHON_FILES)

format:
	@echo "Running ruff for formatting..."
	ruff format $(PYTHON_FILES)
	@echo "Running isort for formatting..."
	isort $(PYTHON_FILES)

all: format lint

clean-experiments:
	@echo "Cleaning experiments..."
	@rm -rf results