# Bootstrap and Setup

## Bootstrap and Setup (When Network Access Available)
- Clone repository: `git clone https://github.com/Clarifai/clarifai-python.git`
- Create virtual environment: `python3 -m venv .venv` -- takes 3 seconds
- Activate environment: `source .venv/bin/activate`
- Install package: `pip install -e .` -- takes 30 seconds. NEVER CANCEL. Set timeout to 2+ minutes.
- Install dev dependencies: `pip install -r requirements-dev.txt` -- takes 10 seconds
- Install pre-commit: `pre-commit install` -- takes under 1 second

## Alternative Setup for Network-Limited Environments
**IMPORTANT**: In environments with network connectivity issues, most pip install commands fail.
- **For Development**: Install package from PyPI in a working environment: `pip install clarifai`
- **For Testing**: Use working virtual environment or global install if available
- **Code Changes**: Use `ruff check . --fix` and `ruff format .` which work without installation
- **Import Testing**: Requires either global clarifai install or working virtual environment

## CI-Preferred UV-based Setup (When Available)
- Upgrade pip: `python -m pip install --upgrade pip`
- Install uv: `pip install uv` -- takes 10 seconds
- Create venv: `uv venv` -- takes 5 seconds
- Install dependencies: `uv pip install -r requirements.txt -r tests/requirements.txt` -- takes 60 seconds. NEVER CANCEL. Set timeout to 3+ minutes.

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
