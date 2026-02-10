"""Tests for the login and create-context command improvements."""

import os
from unittest import mock

from click.testing import CliRunner

from clarifai.cli.base import cli


class TestLoginCommand:
    """Test cases for the login command."""

    def setup_method(self):
        self.runner = CliRunner()
        self.validate_patch = mock.patch('clarifai.utils.cli.validate_context_auth')
        self.mock_validate = self.validate_patch.start()

    def teardown_method(self):
        self.validate_patch.stop()

    def test_login_with_env_var_accepted(self):
        """Test login when CLARIFAI_PAT env var exists and user accepts it."""
        with self.runner.isolated_filesystem():
            with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat_123'}):
                result = self.runner.invoke(
                    cli, ['login'], input='testuser\ny\n', catch_exceptions=False
                )

        assert result.exit_code == 0
        assert 'Use CLARIFAI_PAT from environment?' in result.output
        assert "Success! You're logged in as testuser" in result.output
        # Should not prompt for context name (auto-creates 'default')
        assert 'Enter a name for this context' not in result.output
        # Should not show verbose context explanation from old flow
        assert "Let's save these credentials to a new context" not in result.output
        # Validation debug logs should not leak into output
        assert 'Validating the Context Credentials' not in result.output

    def test_login_with_env_var_declined(self):
        """Test login when user declines to use CLARIFAI_PAT env var."""
        with self.runner.isolated_filesystem():
            with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat_123'}):
                with mock.patch('clarifai.cli.base.masked_input', return_value='manual_pat'):
                    result = self.runner.invoke(
                        cli, ['login'], input='testuser\nn\n', catch_exceptions=False
                    )

        assert result.exit_code == 0
        assert 'Use CLARIFAI_PAT from environment?' in result.output
        assert 'Create a PAT at:' in result.output
        assert "Success! You're logged in as testuser" in result.output

    def test_login_without_env_var(self):
        """Test login when CLARIFAI_PAT env var is not set."""
        with self.runner.isolated_filesystem():
            env = os.environ.copy()
            env.pop('CLARIFAI_PAT', None)

            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch('clarifai.cli.base.masked_input', return_value='typed_pat') as mock_masked:
                    result = self.runner.invoke(
                        cli, ['login'], input='testuser\n', catch_exceptions=False
                    )

        assert result.exit_code == 0
        assert "you'll need a Personal Access Token (PAT)" in result.output
        assert 'Create one at:' in result.output
        assert 'Set CLARIFAI_PAT environment variable to skip this prompt' in result.output
        assert "Success! You're logged in as testuser" in result.output
        mock_masked.assert_called_once()

    def test_login_with_user_id_option(self):
        """Test login with --user_id skips the user ID prompt."""
        with self.runner.isolated_filesystem():
            with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
                result = self.runner.invoke(
                    cli, ['login', '--user_id', 'presetuser'], input='y\n', catch_exceptions=False
                )

        assert result.exit_code == 0
        assert 'Enter your Clarifai user ID' not in result.output
        assert "Success! You're logged in as presetuser" in result.output


class TestCreateContextCommand:
    """Test cases for the create-context command improvements."""

    def setup_method(self):
        self.runner = CliRunner()
        self.validate_patch = mock.patch('clarifai.utils.cli.validate_context_auth')
        self.mock_validate = self.validate_patch.start()

    def teardown_method(self):
        self.validate_patch.stop()

    def _login_first(self, config_path='./config.yaml'):
        """Helper to create initial config via login."""
        with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
            self.runner.invoke(
                cli, ['--config', config_path, 'login'], input='testuser\ny\n'
            )

    def test_create_context_with_env_var(self):
        """Test create-context detects and offers CLARIFAI_PAT from environment."""
        with self.runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                self._login_first()

                with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'another_pat'}):
                    result = self.runner.invoke(
                        cli,
                        ['--config', './config.yaml', 'config', 'create-context', 'dev'],
                        input='devuser\nhttps://api-dev.clarifai.com\ny\n',
                        catch_exceptions=False,
                    )

        assert result.exit_code == 0
        assert 'Found CLARIFAI_PAT in environment. Use it?' in result.output

    def test_create_context_without_env_var(self):
        """Test create-context uses masked_input when no env var is set."""
        with self.runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                self._login_first()

                env = os.environ.copy()
                env.pop('CLARIFAI_PAT', None)

                with mock.patch.dict(os.environ, env, clear=True):
                    with mock.patch('clarifai.cli.base.masked_input', return_value='new_pat') as mock_masked:
                        result = self.runner.invoke(
                            cli,
                            ['--config', './config.yaml', 'config', 'create-context', 'prod'],
                            input='produser\nhttps://api.clarifai.com\n',
                            catch_exceptions=False,
                        )

        assert result.exit_code == 0
        assert 'Set CLARIFAI_PAT environment variable' in result.output
        mock_masked.assert_called_once()
