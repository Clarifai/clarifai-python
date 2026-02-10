"""Tests for the login command improvements."""

import os
from unittest import mock

import pytest
from click.testing import CliRunner

from clarifai.cli.base import cli


class TestLoginCommand:
    """Test cases for the login command."""

    @pytest.fixture
    def runner(self):
        """Provide a CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def mock_validate(self):
        """Mock the validate_context_auth function."""
        with mock.patch('clarifai.utils.cli.validate_context_auth') as mock_val:
            yield mock_val

    def test_login_with_env_var_accepts(self, runner, mock_validate):
        """Test login when CLARIFAI_PAT env var exists and user accepts it."""
        with runner.isolated_filesystem():
            with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat_123'}):
                result = runner.invoke(
                    cli, ['login'], input='testuser\ny\n', catch_exceptions=False
                )

            assert result.exit_code == 0
            assert 'Use CLARIFAI_PAT from environment?' in result.output
            assert "Success! You're logged in as testuser" in result.output
            assert 'manage multiple accounts or environments' in result.output
            # Should NOT show PAT creation instructions when using env var
            assert 'Create one at:' not in result.output or result.output.index(
                'Use CLARIFAI_PAT'
            ) < result.output.index('Create one at:')

    def test_login_with_env_var_declines(self, runner, mock_validate):
        """Test login when user declines to use CLARIFAI_PAT env var."""
        with runner.isolated_filesystem():
            with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat_123'}):
                with mock.patch('clarifai.cli.base.masked_input', return_value='manual_pat'):
                    result = runner.invoke(
                        cli, ['login'], input='testuser\nn\n', catch_exceptions=False
                    )

            assert result.exit_code == 0
            assert 'Use CLARIFAI_PAT from environment?' in result.output
            # Should show PAT creation instructions when user declines env var
            assert 'Create a PAT at:' in result.output
            assert "Success! You're logged in as testuser" in result.output

    def test_login_without_env_var(self, runner, mock_validate):
        """Test login when CLARIFAI_PAT env var is not set."""
        with runner.isolated_filesystem():
            # Ensure CLARIFAI_PAT is not in environment
            env = os.environ.copy()
            env.pop('CLARIFAI_PAT', None)

            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch('clarifai.cli.base.masked_input', return_value='typed_pat'):
                    result = runner.invoke(
                        cli, ['login'], input='testuser\n', catch_exceptions=False
                    )

            assert result.exit_code == 0
            # Should show full PAT instructions when no env var
            assert "you'll need a Personal Access Token (PAT)" in result.output
            assert 'Create one at:' in result.output
            assert 'Set CLARIFAI_PAT environment variable to skip this prompt' in result.output
            assert "Success! You're logged in as testuser" in result.output

    def test_login_auto_creates_default_context(self, runner, mock_validate):
        """Test that login automatically creates 'default' context without asking."""
        with runner.isolated_filesystem():
            with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
                result = runner.invoke(
                    cli, ['login'], input='testuser\ny\n', catch_exceptions=False
                )

            assert result.exit_code == 0
            # Should NOT prompt for context name
            assert 'Enter a name for this context' not in result.output
            # Should show success message
            assert "Success! You're logged in as testuser" in result.output

    def test_login_success_message_format(self, runner, mock_validate):
        """Test that success message has correct format."""
        with runner.isolated_filesystem():
            with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
                result = runner.invoke(
                    cli, ['login'], input='myuser\ny\n', catch_exceptions=False
                )

            assert result.exit_code == 0
            # Check for new simplified success message
            assert "Success! You're logged in as myuser" in result.output
            # Check for tip about managing contexts
            assert '💡 Tip:' in result.output
            assert 'clarifai config' in result.output
            assert 'multiple accounts or environments' in result.output

    def test_login_no_verbose_context_explanation(self, runner, mock_validate):
        """Test that login doesn't show verbose context explanation."""
        with runner.isolated_filesystem():
            with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
                result = runner.invoke(
                    cli, ['login'], input='testuser\ny\n', catch_exceptions=False
                )

            assert result.exit_code == 0
            # Should NOT show verbose context explanation
            assert "Let's save these credentials to a new context" not in result.output
            assert (
                'You can have multiple contexts to easily switch between accounts or projects'
                not in result.output
            )

    def test_login_with_user_id_option(self, runner, mock_validate):
        """Test login with --user_id option provided."""
        with runner.isolated_filesystem():
            with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
                result = runner.invoke(
                    cli, ['login', '--user_id', 'presetuser'], input='y\n', catch_exceptions=False
                )

            assert result.exit_code == 0
            # Should not prompt for user_id
            assert 'Enter your Clarifai user ID' not in result.output
            assert "Success! You're logged in as presetuser" in result.output

    def test_login_validates_token(self, runner, mock_validate):
        """Test that login calls validate_context_auth."""
        with runner.isolated_filesystem():
            with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
                result = runner.invoke(
                    cli, ['login'], input='testuser\ny\n', catch_exceptions=False
                )

            assert result.exit_code == 0
            # Verify validation was called
            mock_validate.assert_called_once()
            # Check that it shows verification progress
            assert 'Verifying token' in result.output

    def test_login_masked_input_called_when_needed(self, runner, mock_validate):
        """Test that masked_input is called when user types PAT manually."""
        with runner.isolated_filesystem():
            env = os.environ.copy()
            env.pop('CLARIFAI_PAT', None)

            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch(
                    'clarifai.cli.base.masked_input', return_value='typed_pat'
                ) as mock_masked:
                    result = runner.invoke(
                        cli, ['login'], input='testuser\n', catch_exceptions=False
                    )

            assert result.exit_code == 0
            # Verify masked_input was called
            mock_masked.assert_called_once()
            assert 'Enter your Personal Access Token (PAT):' in mock_masked.call_args[0][0]

    def test_login_no_validation_logs_by_default(self, runner, mock_validate):
        """Test that validation debug logs are not shown by default."""
        with runner.isolated_filesystem():
            with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
                result = runner.invoke(
                    cli, ['login'], input='testuser\ny\n', catch_exceptions=False
                )

            assert result.exit_code == 0
            # Should NOT show validation debug messages
            assert 'Validating the Context Credentials' not in result.output
            assert 'Context is valid' not in result.output
            # Should show clean success message
            assert "Success! You're logged in" in result.output


class TestCreateContextCommand:
    """Test cases for the create-context command improvements."""

    @pytest.fixture
    def runner(self):
        """Provide a CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def mock_validate(self):
        """Mock the validate_context_auth function."""
        with mock.patch('clarifai.utils.cli.validate_context_auth') as mock_val:
            yield mock_val

    def test_create_context_with_env_var(self, runner, mock_validate):
        """Test create-context when CLARIFAI_PAT env var exists."""
        with runner.isolated_filesystem():
            # Mock the config path to use isolated filesystem
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                # First create config with initial login
                with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
                    runner.invoke(cli, ['--config', './config.yaml', 'login'], input='testuser\ny\n')

                # Now test creating a new context
                with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'another_pat'}):
                    result = runner.invoke(
                        cli,
                        ['--config', './config.yaml', 'config', 'create-context', 'dev'],
                        input='devuser\nhttps://api-dev.clarifai.com\ny\n',
                        catch_exceptions=False,
                    )

                assert result.exit_code == 0
                assert 'Found CLARIFAI_PAT in environment. Use it?' in result.output

    def test_create_context_without_env_var(self, runner, mock_validate):
        """Test create-context when CLARIFAI_PAT env var is not set."""
        with runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                # First create config with initial login
                with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
                    runner.invoke(cli, ['--config', './config.yaml', 'login'], input='testuser\ny\n')

                # Now test creating a new context without env var
                env = os.environ.copy()
                env.pop('CLARIFAI_PAT', None)

                with mock.patch.dict(os.environ, env, clear=True):
                    with mock.patch('clarifai.cli.base.masked_input', return_value='new_pat'):
                        result = runner.invoke(
                            cli,
                            ['--config', './config.yaml', 'config', 'create-context', 'prod'],
                            input='produser\nhttps://api.clarifai.com\n',
                            catch_exceptions=False,
                        )

                assert result.exit_code == 0
                assert 'Set CLARIFAI_PAT environment variable' in result.output

    def test_create_context_uses_masked_input(self, runner, mock_validate):
        """Test that create-context uses masked_input for PAT entry."""
        with runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                # First create config with initial login
                with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
                    runner.invoke(cli, ['--config', './config.yaml', 'login'], input='testuser\ny\n')

                # Now test creating a new context
                env = os.environ.copy()
                env.pop('CLARIFAI_PAT', None)

                with mock.patch.dict(os.environ, env, clear=True):
                    with mock.patch(
                        'clarifai.cli.base.masked_input', return_value='secure_pat'
                    ) as mock_masked:
                        result = runner.invoke(
                            cli,
                            ['--config', './config.yaml', 'config', 'create-context', 'staging'],
                            input='staginguser\nhttps://api-staging.clarifai.com\n',
                            catch_exceptions=False,
                        )

                assert result.exit_code == 0
                # Verify masked_input was called
                mock_masked.assert_called_once()
