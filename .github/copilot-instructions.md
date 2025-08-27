# Clarifai Python SDK

Clarifai Python SDK is the official Python client for the Clarifai AI platform, providing computer vision and natural language processing capabilities. The SDK includes both a Python API and a CLI tool for model operations, compute orchestration, and data management.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Setup (When Network Access Available)
- Clone repository: `git clone https://github.com/Clarifai/clarifai-python.git`
- Create virtual environment: `python3 -m venv .venv` -- takes 3 seconds
- Activate environment: `source .venv/bin/activate`
- Install package: `pip install -e .` -- takes 30 seconds. NEVER CANCEL. Set timeout to 2+ minutes.
- Install dev dependencies: `pip install -r requirements-dev.txt` -- takes 10 seconds
- Install pre-commit: `pre-commit install` -- takes under 1 second

### Alternative Setup for Network-Limited Environments
**IMPORTANT**: In environments with network connectivity issues, most pip install commands fail.
- **For Development**: Install package from PyPI in a working environment: `pip install clarifai`
- **For Testing**: Use working virtual environment or global install if available
- **Code Changes**: Use `ruff check . --fix` and `ruff format .` which work without installation
- **Import Testing**: Requires either global clarifai install or working virtual environment

### CI-Preferred UV-based Setup (When Available)
- Upgrade pip: `python -m pip install --upgrade pip`
- Install uv: `pip install uv` -- takes 10 seconds
- Create venv: `uv venv` -- takes 5 seconds
- Install dependencies: `uv pip install -r requirements.txt -r tests/requirements.txt` -- takes 60 seconds. NEVER CANCEL. Set timeout to 3+ minutes.

### Build and Test (Core Development Tools)
- **Lint code**: `ruff check . --fix` -- takes under 1 second. WORKS WITHOUT INSTALLATION.
- **Format code**: `ruff format --check .` -- takes under 1 second. WORKS WITHOUT INSTALLATION.
- **Pre-commit**: `pre-commit run --all-files` -- may fail due to network issues during initial setup
- **Tests**: `pytest` -- requires full installation and test dependencies

### Validation Scenarios
After making changes, ALWAYS test these core scenarios (requires working clarifai installation):
- **Basic Import Test**: `python -c "import clarifai; print(f'Version: {clarifai.__version__}')"`
- **CLI Functionality**: `clarifai --help` should show all available commands
- **CLI Commands**: Test `clarifai model --help`, `clarifai pipeline --help`
- **Model Operations**: If changing model code, test help commands and basic imports
- **Pipeline Operations**: If changing pipeline code, test help commands and basic imports

**Linting/Formatting (works without installation):**
- **Lint Check**: `ruff check . --fix` -- identifies and fixes code style issues
- **Format Check**: `ruff format --check .` -- checks code formatting

## Installation Requirements and Limitations

### Network Dependencies - CRITICAL LIMITATIONS
- **MAJOR ISSUE**: Many environments have PyPI connectivity timeouts causing all pip installs to fail with `ReadTimeoutError`
- **pip install -e .** may fail with network timeouts in fresh environments
- **pip install uv** may fail with network timeouts
- **pip install -r requirements-dev.txt** may fail with network timeouts
- **pre-commit install** may fail during hook setup due to network timeouts
- **Solution**: Use direct import method and document network limitation: "pip install commands fail due to network/firewall limitations"

### Python Version Support
- Supports Python 3.9, 3.10, 3.11, 3.12
- CI primarily tests Python 3.11 and 3.12
- Uses `setup.py` and `pyproject.toml` for packaging

### Build Times and Timeouts
- **NEVER CANCEL**: Basic installation takes 30-60 seconds
- **NEVER CANCEL**: Full test suite takes 5-15 minutes
- **NEVER CANCEL**: Pre-commit setup can take 2+ minutes
- Always set timeouts to at least double the expected time

## Key Projects and Code Structure

### Main Directories
- `clarifai/` - Main SDK source code
  - `clarifai/cli/` - Command-line interface implementation
  - `clarifai/client/` - API client classes (User, App, Model, etc.)
  - `clarifai/runners/` - Model and pipeline step builders
  - `clarifai/datasets/` - Dataset management utilities
  - `clarifai/rag/` - Retrieval Augmented Generation features
- `tests/` - Test suite with unit and integration tests
- `scripts/` - Utility scripts for development
- `.github/workflows/` - CI/CD pipeline definitions

### Key Entry Points
- **CLI Tool**: `clarifai` command (defined in `clarifai.cli.base:cli`)
- **Python API**: Import with `from clarifai.client.user import User`
- **Model Operations**: `clarifai.client.model.Model`
- **Workflow Operations**: `clarifai.client.workflow.Workflow`

### Configuration Files
- `.pre-commit-config.yaml` - Pre-commit hook configuration using ruff
- `.ruff.toml` - Ruff linter and formatter settings
- `pyproject.toml` - Modern Python project configuration
- `setup.py` - Traditional Python package setup
- `requirements.txt` - Core package dependencies
- `requirements-dev.txt` - Development-only dependencies

## CLI Operations

### Available Commands
```bash
clarifai --help                    # Show all available commands
clarifai model --help             # Model operations (list, predict, upload, etc.)
clarifai pipeline --help          # Pipeline operations
clarifai computecluster --help    # Compute cluster management
clarifai nodepool --help          # Nodepool management
clarifai deployment --help        # Deployment management
clarifai login --help             # Authentication setup
```

### Authentication
- Requires `CLARIFAI_PAT` environment variable or config file
- Use `clarifai login` to configure authentication
- Test commands may require valid API credentials

## Common Development Tasks

### Making Code Changes
1. **Linting**: `ruff check . --fix` (requires ruff installation or working venv)
2. **Formatting**: `ruff format .` (requires ruff installation or working venv)
3. **Basic validation**: `python -c "import clarifai"` (requires working installation)
4. **CLI testing**: `clarifai --help` (requires working installation)
5. **Run targeted tests**: `pytest tests/path/to/relevant/test.py` (requires test setup)

### Before Committing (When Tools Available)
- **ALWAYS** run `ruff check . --fix` before committing or CI will fail
- **ALWAYS** run `ruff format .` before committing or CI will fail
- Test that basic imports still work: `python -c "import clarifai"`
- If changing CLI code, test: `clarifai --help`

### Working Without Full Installation
**When network issues prevent installation:**
1. **Code Review**: Use file viewing and git diff to review changes
2. **Documentation**: Read README.md, docstrings, and configuration files
3. **File Structure**: Explore `clarifai/` directory structure and module organization
4. **Configuration**: Check `.ruff.toml`, `pyproject.toml`, setup.py for project settings
5. **CI Understanding**: Review `.github/workflows/` to understand build/test process

### Model and Pipeline Development
- **Model builder**: Located in `clarifai/runners/models/model_builder.py`
- **Pipeline builder**: Located in `clarifai/runners/pipeline_steps/pipeline_step_builder.py`
- **Local testing commands**: `clarifai model local-test`, `clarifai pipeline-step test`
- **Docker support**: Available for containerized model testing (requires Docker)
- **Configuration files**: YAML-based configs for models, pipelines, deployments

### Troubleshooting

**Installation Issues:**
- **Import errors**: Ensure `pip install -e .` completed successfully or use global clarifai installation
- **CLI not found**: Ensure virtual environment is activated and package installed
- **Network timeouts**: Common issue; document as environment limitation
- **"ModuleNotFoundError: google"**: Missing protobuf dependencies; requires successful `pip install -e .`

**Development Issues:**
- **Test failures**: May require `CLARIFAI_PAT` environment variable for API tests
- **Lint failures**: Use `ruff check . --fix` to auto-fix most style issues
- **CLI authentication**: Set up with `clarifai login` or `CLARIFAI_PAT` environment variable
- **Import issues after changes**: Reinstall with `pip install -e .` to pick up changes

**Common Error Messages:**
- `ReadTimeoutError: HTTPSConnectionPool(host='pypi.org')`: Network/firewall limitation
- `error: subprocess-exited-with-error` during pip install: Network connectivity issue
- `ModuleNotFoundError`: Missing dependencies due to incomplete installation
- `TimeoutError: The read operation timed out`: PyPI connectivity problem

## Frequently Used Commands Output

### Repository Root
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

### Package Information
```
$ python -c "import clarifai; print(f'Version: {clarifai.__version__}')"
Clarifai imported successfully
Version: 11.6.8
```

### CLI Help Output
```
$ clarifai --help
Usage: clarifai [OPTIONS] COMMAND [ARGS]...

  Clarifai CLI

Commands:
  computecluster (cc)     Manage Compute Clusters
  config                  Manage configuration profiles
  deployment (dp)         Manage Deployments
  login                   Login command to set PAT
  model                   Manage & Develop Models
  nodepool (np)           Manage Nodepools
  pipeline (pl)           Manage pipelines
  pipeline-step (ps)      Manage pipeline steps
  run                     Execute script with context
  shell-completion        Shell completion script
```
