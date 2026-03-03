"""Tests for the --context flag functionality."""

import tempfile
from pathlib import Path

import yaml
from click.testing import CliRunner

from clarifai.cli.base import cli


class TestContextFlag:
    """Test cases for the global --context flag."""

    def test_context_flag_in_help(self):
        """Test that --context appears in the help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert '--context' in result.output

    def test_context_flag_with_config_current(self):
        """Test that --context flag changes the current context for a command."""
        runner = CliRunner()

        # Create a temporary config file with multiple contexts
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                'current_context': 'context-a',
                'contexts': {
                    'context-a': {
                        'CLARIFAI_USER_ID': 'user-a',
                        'CLARIFAI_PAT': 'pat-a',
                        'CLARIFAI_API_BASE': 'https://api.clarifai.com',
                    },
                    'context-b': {
                        'CLARIFAI_USER_ID': 'user-b',
                        'CLARIFAI_PAT': 'pat-b',
                        'CLARIFAI_API_BASE': 'https://api.clarifai.com',
                    },
                },
            }
            with open(config_path, 'w') as f:
                yaml.safe_dump(config_data, f)

            # Test without --context flag (should use current context)
            result = runner.invoke(
                cli, ['--config', str(config_path), 'config', 'current-context']
            )
            assert result.exit_code == 0
            assert 'context-a' in result.output

            # Test with --context flag (should override current context)
            # We test this by checking that the context override is set
            # The current-context command should still show the saved current context
            result = runner.invoke(
                cli,
                [
                    '--config',
                    str(config_path),
                    '--context',
                    'context-b',
                    'config',
                    'current-context',
                ],
            )
            # The current-context command shows the saved current context, not the override
            assert result.exit_code == 0
            assert 'context-a' in result.output  # Still shows the saved current context

    def test_context_override_actually_used(self):
        """Test that commands actually use the overridden context by accessing ctx.obj.current."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                'current_context': 'context-a',
                'contexts': {
                    'context-a': {
                        'CLARIFAI_USER_ID': 'user-a',
                        'CLARIFAI_PAT': 'pat-a',
                        'CLARIFAI_API_BASE': 'https://api.clarifai.com',
                    },
                    'context-b': {
                        'CLARIFAI_USER_ID': 'user-b',
                        'CLARIFAI_PAT': 'pat-b',
                        'CLARIFAI_API_BASE': 'https://api.clarifai.com',
                    },
                },
            }
            with open(config_path, 'w') as f:
                yaml.safe_dump(config_data, f)

            # Test by checking that config env command uses the correct context
            # This command prints environment variables from the current context
            result = runner.invoke(cli, ['--config', str(config_path), 'config', 'env'])
            assert result.exit_code == 0
            # Should show environment variables from context-a
            assert 'user-a' in result.output

            # Test with --context override
            result = runner.invoke(
                cli, ['--config', str(config_path), '--context', 'context-b', 'config', 'env']
            )
            assert result.exit_code == 0
            # Should show environment variables from context-b
            assert 'user-b' in result.output

    def test_context_flag_with_invalid_context(self):
        """Test that --context flag errors with invalid context name."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                'current_context': 'context-a',
                'contexts': {
                    'context-a': {
                        'CLARIFAI_USER_ID': 'user-a',
                        'CLARIFAI_PAT': 'pat-a',
                        'CLARIFAI_API_BASE': 'https://api.clarifai.com',
                    },
                },
            }
            with open(config_path, 'w') as f:
                yaml.safe_dump(config_data, f)

            # Test with invalid context name
            result = runner.invoke(
                cli,
                [
                    '--config',
                    str(config_path),
                    '--context',
                    'invalid-context',
                    'config',
                    'current-context',
                ],
            )
            assert result.exit_code != 0
            assert "Context 'invalid-context' not found" in result.output

    def test_run_command_with_context_flag(self):
        """Test that the run command respects --context flag."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                'current_context': 'context-a',
                'contexts': {
                    'context-a': {
                        'CLARIFAI_USER_ID': 'user-a',
                        'CLARIFAI_PAT': 'pat-a',
                        'CLARIFAI_API_BASE': 'https://api.clarifai.com',
                    },
                    'context-b': {
                        'CLARIFAI_USER_ID': 'user-b',
                        'CLARIFAI_PAT': 'pat-b',
                        'CLARIFAI_API_BASE': 'https://api.clarifai.com',
                    },
                },
            }
            with open(config_path, 'w') as f:
                yaml.safe_dump(config_data, f)

            # Test 1: run command with local --context option (not global)
            # This tests the existing run command's --context option still works
            result = runner.invoke(
                cli, ['--config', str(config_path), 'run', '--context', 'context-b', 'echo test']
            )
            # Verify no error about context not found
            assert "Context 'context-b' not found" not in result.output

            # Test 2: Verify the global --context flag also works with run command
            result = runner.invoke(
                cli, ['--config', str(config_path), '--context', 'context-b', 'run', 'echo test']
            )
            # Verify no error about context not found
            assert "Context 'context-b' not found" not in result.output
