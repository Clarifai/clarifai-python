# Common Development Tasks

## Making Code Changes
1. **Linting**: `ruff check . --fix` (requires ruff installation or working venv)
2. **Formatting**: `ruff format .` (requires ruff installation or working venv)
3. **Basic validation**: `python -c "import clarifai"` (requires working installation)
4. **CLI testing**: `clarifai --help` (requires working installation)
5. **Run targeted tests**: `pytest tests/path/to/relevant/test.py` (requires test setup)

## Working Without Full Installation
**When network issues prevent installation:**
1. **Code Review**: Use file viewing and git diff to review changes
2. **Documentation**: Read README.md, docstrings, and configuration files
3. **File Structure**: Explore `clarifai/` directory structure and module organization
4. **Configuration**: Check `.ruff.toml`, `pyproject.toml`, setup.py for project settings
5. **CI Understanding**: Review `.github/workflows/` to understand build/test process

## Model and Pipeline Development
- **Model builder**: Located in `clarifai/runners/models/model_builder.py`
- **Pipeline builder**: Located in `clarifai/runners/pipeline_steps/pipeline_step_builder.py`
- **Local testing commands**: `clarifai model local-test`, `clarifai pipeline-step test`
- **Docker support**: Available for containerized model testing (requires Docker)
- **Configuration files**: YAML-based configs for models, pipelines, deployments
