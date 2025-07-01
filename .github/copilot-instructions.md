This is a Python based repository with a Python client for the Clarifai API. Please follow these
guidelines when contributing:

## Code Standards

### Required Before Each Commit
- Run `pre-commit run -a` before committing any changes to ensure proper code formatting and linting
- This will run ruff on all Python files to maintain consistent style.
- Fix any linter / formatting errors that are returned. This may require running multiple times.

### Development Flow
- Install dependencies: `(curl -LsSf https://astral.sh/uv/install.sh | sh) && uv venv && uv pip install -r requirements.txt -r tests/requirements.txt && pre-commit install`
- Test: `export PYTHONPATH=. && export CLARIFAI_USER_ID="$(uv run python scripts/key_for_tests.py --get-userid)" && export CLARIFAI_PAT="$(uv run python scripts/key_for_tests.py --create-pat)" && uv run pytest --cov=. --cov-report=xml:coverage/coverage.cobertura.xml --ignore=tests/runners/test_model_run_locally-container.py`

## Repository Structure
- `tests/`: Where all the tests are
- `clarifai/`: The main Clarifai API client packages

## Key Guidelines
1. Follow Python best practices and idiomatic patterns
2. Maintain existing code structure and organization
3. Use dependency injection patterns where appropriate
4. Write unit tests for new functionality. Use table-driven unit tests when possible.
5. Document public APIs and complex logic. Suggest changes to the `README.md` folder when appropriate
