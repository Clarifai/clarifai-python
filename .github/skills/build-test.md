# Build and Test

## Core Development Tools
- **Lint code**: `ruff check . --fix` -- takes under 1 second. WORKS WITHOUT INSTALLATION.
- **Format code**: `ruff format --check .` -- takes under 1 second. WORKS WITHOUT INSTALLATION.
- **Pre-commit**: `pre-commit run --all-files` -- may fail due to network issues during initial setup
- **Tests**: `pytest` -- requires full installation and test dependencies

## Validation Scenarios
After making changes, ALWAYS test these core scenarios (requires working clarifai installation):
- **Basic Import Test**: `python -c "import clarifai; print(f'Version: {clarifai.__version__}')"`
- **CLI Functionality**: `clarifai --help` should show all available commands
- **CLI Commands**: Test `clarifai model --help`, `clarifai pipeline --help`
- **Model Operations**: If changing model code, test help commands and basic imports
- **Pipeline Operations**: If changing pipeline code, test help commands and basic imports

## Linting/Formatting (works without installation)
- **Lint Check**: `ruff check . --fix` -- identifies and fixes code style issues
- **Format Check**: `ruff format --check .` -- checks code formatting

## Before Committing (When Tools Available)
- **ALWAYS** run `ruff check . --fix` before committing or CI will fail
- **ALWAYS** run `ruff format .` before committing or CI will fail
- Test that basic imports still work: `python -c "import clarifai"`
- If changing CLI code, test: `clarifai --help`
