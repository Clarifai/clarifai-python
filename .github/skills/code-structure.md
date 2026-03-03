# Code Structure

## Main Directories
- `clarifai/` - Main SDK source code
  - `clarifai/cli/` - Command-line interface implementation
  - `clarifai/client/` - API client classes (User, App, Model, etc.)
  - `clarifai/runners/` - Model and pipeline step builders
  - `clarifai/datasets/` - Dataset management utilities
  - `clarifai/rag/` - Retrieval Augmented Generation features
- `tests/` - Test suite with unit and integration tests
- `scripts/` - Utility scripts for development
- `.github/workflows/` - CI/CD pipeline definitions

## Key Entry Points
- **CLI Tool**: `clarifai` command (defined in `clarifai.cli.base:cli`)
- **Python API**: Import with `from clarifai.client.user import User`
- **Model Operations**: `clarifai.client.model.Model`
- **Workflow Operations**: `clarifai.client.workflow.Workflow`

## Configuration Files
- `.pre-commit-config.yaml` - Pre-commit hook configuration using ruff
- `.ruff.toml` - Ruff linter and formatter settings
- `pyproject.toml` - Modern Python project configuration
- `setup.py` - Traditional Python package setup
- `requirements.txt` - Core package dependencies
- `requirements-dev.txt` - Development-only dependencies

## Repository Root
```
$ ls -la
.coveragerc
.git/
.github/
.gitignore
.isort.cfg
.pre-commit-config.yaml
.ruff.toml
CHANGELOG.md
LICENSE
MANIFEST.in
README.md
clarifai/
pyproject.toml
pytest.ini
requirements-dev.txt
requirements.txt
scripts/
setup.py
tests/
```
