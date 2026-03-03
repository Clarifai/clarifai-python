# Agent Skills

This directory contains specialized skills for working with the Clarifai Python SDK. Each skill file focuses on a specific aspect of development.

## Available Skills

### bootstrap-setup.md
Environment setup, installation, and configuration instructions including:
- Standard setup with pip
- Alternative setup for network-limited environments
- UV-based setup for CI
- Installation requirements and limitations
- Python version support

### build-test.md
Build, lint, format, and test instructions including:
- Core development tools (ruff, pytest)
- Validation scenarios
- Pre-commit checks
- Linting and formatting commands

### code-structure.md
Repository structure and key files including:
- Main directories and their purposes
- Key entry points (CLI and Python API)
- Configuration files
- Repository layout

### cli-operations.md
CLI commands and usage including:
- Available commands
- Authentication setup
- CLI help output
- Package information

### development-tasks.md
Common development workflows including:
- Making code changes
- Working without full installation
- Model and pipeline development
- Testing procedures

### troubleshooting.md
Solutions to common issues including:
- Installation issues
- Development issues
- Common error messages
- Network and dependency problems

## Usage

These skills are designed to be used by AI coding assistants to help developers work more effectively with the Clarifai Python SDK. Each skill can be referenced individually based on the task at hand.

## Symlinks

The following directories contain symlinks to this skills directory:
- `.claude/skills` → `.github/skills`
- `.gemini/skills` → `.github/skills`
- `.codex/skills` → `.github/skills`

This allows different AI assistants to access the same set of skills while maintaining their own directory structures.
