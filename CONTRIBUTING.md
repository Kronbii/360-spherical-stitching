# Contributing Guide

Thank you for your interest in contributing to this project.

## Getting Started

1. Fork the repository and create a feature branch from `main`.
2. Set up a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Development Workflow

1. Make focused changes in small commits.
2. Add or update tests when behavior changes.
3. Run tests before opening a pull request:

```bash
python run_tests.py
```

You can also run targeted tests:

```bash
python run_tests.py tests/test_config.py
python run_tests.py -m "not slow"
```

## Code Style

- Follow existing code style and naming conventions in the repository.
- Keep functions and modules focused and readable.
- Add docstrings/comments when they improve maintainability.
- Avoid unrelated refactors in the same pull request.

## Commit Messages

- Use clear, imperative commit titles.
- Keep each commit scoped to a single logical change.
- Include context in the body when needed.

Example:

```
Improve homography outlier rejection on low-texture frames
```

## Pull Request Checklist

Before submitting a pull request, make sure:

- Tests pass locally.
- New behavior is covered by tests or clearly justified.
- Documentation is updated (`README.md`, `USAGE.md`, `TECHNICAL.md`) when needed.
- The change is backward compatible, or the breaking impact is described.

## Reporting Bugs and Requesting Features

- Use the issue templates to report bugs or request features.
- Include reproducible steps, expected behavior, and environment details.
- For security-sensitive issues, follow `SECURITY.md` instead of opening a
  public issue.

## Code of Conduct

By participating, you agree to abide by the [Code of
Conduct](CODE_OF_CONDUCT.md).
