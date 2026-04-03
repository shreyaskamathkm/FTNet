# Contributing to FTNet

Thank you for your interest in contributing! This document covers how to set up your development environment, the branch strategy, code standards, and how to submit changes.

## Table of Contents

- [Branch Strategy](#branch-strategy)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Running Tests](#running-tests)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, published results matching the IEEE Access paper |
| `develop` | Active development — all PRs target this branch |

**All pull requests must target `develop`, not `main`.**

`main` is only updated via a release PR from `develop` when a new stable milestone is reached.

---

## Development Setup

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)
- Git

### Install

```bash
git clone https://github.com/shreyaskamathkm/FTNet.git
cd FTNet
git checkout develop

# Install all dependencies including dev tools
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FTNET_PRETRAINED_DIR` | `<repo_root>/pretrained_models` | Override path for pretrained backbone weights |

---

## Code Standards

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [mypy](https://mypy.readthedocs.io/) for static type checking. These are enforced automatically via pre-commit hooks.

### Manual checks

```bash
# Lint
poetry run ruff check .

# Format
poetry run ruff format .

# Type check
poetry run mypy ftnet --ignore-missing-imports
```

### Key conventions

- **Type hints** are required on all public functions and methods.
- **Docstrings** follow Google style (Args / Returns sections).
- **Logging** — use `logger = logging.getLogger(__name__)` instead of `print()`.
- **Config changes** — all new parameters must be added to the appropriate Pydantic dataclass in `ftnet/cfg/cfg_dataclasses.py` with a sensible default.

---

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=ftnet --cov-report=term-missing

# Run a specific test file
poetry run pytest tests/test_loss.py -v
```

Tests live in the `tests/` directory and mirror the `ftnet/` package structure.

---

## Submitting a Pull Request

1. **Branch** off `develop`:
   ```bash
   git checkout develop
   git pull
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** and write tests for any new functionality.

3. **Ensure all checks pass** locally before opening a PR:
   ```bash
   poetry run pre-commit run --all-files
   poetry run pytest
   ```

4. **Open a PR** against `develop` (not `main`). Fill in the pull request template.

5. A maintainer will review your PR. Please respond to review comments promptly.

---

## Reporting Issues

Use the GitHub issue templates:

- **Bug report** — for incorrect behaviour, crashes, or wrong results.
- **Feature request** — for new dataset support, model variants, or tooling improvements.

Please include your OS, Python version, GPU/CUDA version, and a minimal reproducible example where applicable.
