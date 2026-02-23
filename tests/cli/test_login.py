"""Tests for the login and create-context command improvements."""

import os
from unittest import mock

from click.testing import CliRunner

from clarifai.cli.base import cli


def _mock_verify(pat, api_url):
    """Mock _verify_and_resolve_user that returns a fake user_id."""
    return 'testuser'


def _mock_list_orgs_empty(pat, user_id, api_url):
    return []


def _mock_list_orgs_with_orgs(pat, user_id, api_url):
    return [('clarifai', 'Clarifai'), ('openai', 'OpenAI')]


class TestLoginCommand:
    """Test cases for the login command."""

    def setup_method(self):
        self.runner = CliRunner()
        self.verify_patch = mock.patch(
            'clarifai.cli.base._verify_and_resolve_user', side_effect=_mock_verify
        )
        self.orgs_patch = mock.patch(
            'clarifai.cli.base._list_user_orgs', side_effect=_mock_list_orgs_empty
        )
        self.mock_verify = self.verify_patch.start()
        self.mock_orgs = self.orgs_patch.start()

    def teardown_method(self):
        self.verify_patch.stop()
        self.orgs_patch.stop()

    def test_login_with_pat_flag(self):
        """Non-interactive login with --pat flag, no prompts."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli, ['login', '--pat', 'test_pat_123'], catch_exceptions=False
            )

        assert result.exit_code == 0
        assert 'Verifying...' in result.output
        assert 'Logged in as testuser' in result.output
        assert 'context' in result.output and 'testuser' in result.output

    def test_login_with_env_var(self):
        """Login auto-uses CLARIFAI_PAT from environment (no confirm prompt)."""
        with self.runner.isolated_filesystem():
            with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'env_pat_456'}):
                result = self.runner.invoke(cli, ['login'], catch_exceptions=False)

        assert result.exit_code == 0
        assert 'Using PAT from CLARIFAI_PAT environment variable.' in result.output
        assert 'Logged in as testuser' in result.output
        # No confirm prompt — it just uses the env var
        assert 'Use CLARIFAI_PAT from environment?' not in result.output

    def test_login_interactive_prompt(self):
        """Login prompts for PAT when no flag and no env var."""
        with self.runner.isolated_filesystem():
            env = os.environ.copy()
            env.pop('CLARIFAI_PAT', None)

            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch('clarifai.cli.base.masked_input', return_value='typed_pat'):
                    with mock.patch('clarifai.cli.base.webbrowser', create=True):
                        result = self.runner.invoke(cli, ['login'], catch_exceptions=False)

        assert result.exit_code == 0
        assert 'Logged in as testuser' in result.output

    def test_login_with_user_id_flag(self):
        """--user-id skips org selection and uses the given user_id."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli, ['login', '--pat', 'test_pat', '--user-id', 'openai'], catch_exceptions=False
            )

        assert result.exit_code == 0
        assert 'Logged in as openai' in result.output
        assert 'context' in result.output and 'openai' in result.output

    def test_login_with_name_flag(self):
        """--name sets a custom context name."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli,
                ['login', '--pat', 'test_pat', '--name', 'my-custom'],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        assert 'my-custom' in result.output
        assert 'set as active' in result.output

    def test_login_dev_env_context_naming(self):
        """Dev environment URL produces 'dev-{user_id}' context name."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli,
                ['login', 'https://api-dev.clarifai.com', '--pat', 'test_pat'],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        assert 'dev-testuser' in result.output
        assert 'set as active' in result.output

    def test_login_staging_env_context_naming(self):
        """Staging environment URL produces 'staging-{user_id}' context name."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli,
                ['login', 'https://api-staging.clarifai.com', '--pat', 'test_pat'],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        assert 'staging-testuser' in result.output
        assert 'set as active' in result.output

    def test_login_relogin_updates_existing(self):
        """Re-login updates existing context instead of erroring."""
        with self.runner.isolated_filesystem():
            # First login
            self.runner.invoke(cli, ['login', '--pat', 'old_pat'], catch_exceptions=False)
            # Second login (same user_id resolves to same context name)
            result = self.runner.invoke(cli, ['login', '--pat', 'new_pat'], catch_exceptions=False)

        assert result.exit_code == 0
        assert 'Updated' in result.output
        assert 'testuser' in result.output

    def test_login_creates_new_context(self):
        """First login creates a new context."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli,
                ['--config', './test_config.yaml', 'login', '--pat', 'test_pat'],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        assert 'Created' in result.output
        assert 'testuser' in result.output


class TestLoginOrgSelection:
    """Test cases for org selection during login."""

    def setup_method(self):
        self.runner = CliRunner()
        self.verify_patch = mock.patch(
            'clarifai.cli.base._verify_and_resolve_user', side_effect=_mock_verify
        )
        self.orgs_patch = mock.patch(
            'clarifai.cli.base._list_user_orgs', side_effect=_mock_list_orgs_with_orgs
        )
        self.mock_verify = self.verify_patch.start()
        self.mock_orgs = self.orgs_patch.start()

    def teardown_method(self):
        self.verify_patch.stop()
        self.orgs_patch.stop()

    def test_login_shows_org_list(self):
        """When user has orgs, login shows numbered list."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli, ['login', '--pat', 'test_pat'], input='1\n', catch_exceptions=False
            )

        assert result.exit_code == 0
        # Check key parts are present (colors stripped in test runner)
        assert '[1]' in result.output and 'testuser' in result.output
        assert '[2]' in result.output and 'clarifai' in result.output
        assert '[3]' in result.output and 'openai' in result.output
        assert '(personal)' in result.output
        assert '(Clarifai)' in result.output
        assert '(OpenAI)' in result.output

    def test_login_org_selection_default(self):
        """Pressing enter selects personal user (default=1)."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli, ['login', '--pat', 'test_pat'], input='\n', catch_exceptions=False
            )

        assert result.exit_code == 0
        assert 'Logged in as testuser' in result.output

    def test_login_org_selection_by_number(self):
        """Selecting an org by number works."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli, ['login', '--pat', 'test_pat'], input='2\n', catch_exceptions=False
            )

        assert result.exit_code == 0
        assert 'Logged in as clarifai' in result.output
        assert 'clarifai' in result.output and 'set as active' in result.output

    def test_login_user_id_flag_skips_org_prompt(self):
        """--user-id bypasses org selection even when orgs exist."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli,
                ['login', '--pat', 'test_pat', '--user-id', 'openai'],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        assert 'Select user_id' not in result.output
        assert 'Logged in as openai' in result.output


class TestCreateContextCommand:
    """Test cases for the create-context command."""

    def setup_method(self):
        self.runner = CliRunner()
        self.verify_patch = mock.patch(
            'clarifai.cli.base._verify_and_resolve_user', side_effect=_mock_verify
        )
        self.orgs_patch = mock.patch(
            'clarifai.cli.base._list_user_orgs', side_effect=_mock_list_orgs_empty
        )
        self.mock_verify = self.verify_patch.start()
        self.mock_orgs = self.orgs_patch.start()

    def teardown_method(self):
        self.verify_patch.stop()
        self.orgs_patch.stop()

    def _login_first(self, config_path='./config.yaml'):
        """Helper to create initial config via login."""
        self.runner.invoke(
            cli, ['--config', config_path, 'login', '--pat', 'test_pat'], catch_exceptions=False
        )

    def test_create_context_with_pat_flag(self):
        """Create context with all flags — no prompts."""
        with self.runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                self._login_first()
                result = self.runner.invoke(
                    cli,
                    [
                        '--config',
                        './config.yaml',
                        'config',
                        'create-context',
                        'dev',
                        '--pat',
                        'dev_pat',
                        '--user-id',
                        'devuser',
                    ],
                    catch_exceptions=False,
                )

        assert result.exit_code == 0
        assert 'Context' in result.output and 'dev' in result.output
        assert 'created' in result.output

    def test_create_context_with_env_var(self):
        """Create context auto-uses CLARIFAI_PAT from env."""
        with self.runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                self._login_first()
                with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'env_pat'}):
                    result = self.runner.invoke(
                        cli,
                        ['--config', './config.yaml', 'config', 'create-context', 'myctx'],
                        catch_exceptions=False,
                    )

        assert result.exit_code == 0
        assert 'Using PAT from CLARIFAI_PAT environment variable.' in result.output
        assert 'Context' in result.output and 'myctx' in result.output
        assert 'created' in result.output

    def test_create_context_duplicate_name_fails(self):
        """Creating context with existing name fails."""
        with self.runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                self._login_first()
                result = self.runner.invoke(
                    cli,
                    [
                        '--config',
                        './config.yaml',
                        'config',
                        'create-context',
                        'testuser',
                        '--pat',
                        'x',
                    ],
                    catch_exceptions=False,
                )

        assert result.exit_code == 1
        assert 'already exists' in result.output


def _mock_get_user_info(user_id=None):
    """Return a mock response with user profile fields."""
    from unittest.mock import MagicMock

    user = MagicMock()
    user.id = 'testuser'
    user.full_name = 'Test User'
    user.primary_email = 'test@clarifai.com'
    user.company_name = 'Clarifai'
    resp = MagicMock()
    resp.user = user
    return resp


class TestWhoamiCommand:
    """Test cases for the whoami command."""

    def setup_method(self):
        self.runner = CliRunner()

    def _login_first(self, config_path='./config.yaml', pat='test_pat_123'):
        """Helper to login and create a config."""
        with (
            mock.patch('clarifai.cli.base._verify_and_resolve_user', side_effect=_mock_verify),
            mock.patch('clarifai.cli.base._list_user_orgs', side_effect=_mock_list_orgs_empty),
        ):
            self.runner.invoke(
                cli,
                ['--config', config_path, 'login', '--pat', pat],
                catch_exceptions=False,
            )

    def test_whoami_default(self):
        """Default whoami shows user_id and context from local config (no API call)."""
        with self.runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                self._login_first()
                result = self.runner.invoke(
                    cli, ['--config', './config.yaml', 'whoami'], catch_exceptions=False
                )

        assert result.exit_code == 0
        assert 'testuser' in result.output
        assert 'Context:' in result.output

    def test_whoami_with_orgs(self):
        """--orgs shows organization list."""
        with self.runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                self._login_first()
                with mock.patch(
                    'clarifai.cli.base._list_user_orgs', side_effect=_mock_list_orgs_with_orgs
                ):
                    result = self.runner.invoke(
                        cli,
                        ['--config', './config.yaml', 'whoami', '--orgs'],
                        catch_exceptions=False,
                    )

        assert result.exit_code == 0
        assert 'testuser' in result.output
        assert 'Organizations:' in result.output
        assert 'clarifai' in result.output
        assert 'openai' in result.output

    def test_whoami_with_all(self):
        """--all shows full profile including name, email, company, and orgs."""
        with self.runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                self._login_first()
                with (
                    mock.patch(
                        'clarifai.cli.base._list_user_orgs', side_effect=_mock_list_orgs_with_orgs
                    ),
                    mock.patch(
                        'clarifai.client.user.User.get_user_info', side_effect=_mock_get_user_info
                    ),
                ):
                    result = self.runner.invoke(
                        cli,
                        ['--config', './config.yaml', 'whoami', '--all'],
                        catch_exceptions=False,
                    )

        assert result.exit_code == 0
        assert 'testuser' in result.output
        assert 'Test User' in result.output
        assert 'test@clarifai.com' in result.output
        assert 'Clarifai' in result.output
        assert 'Organizations:' in result.output

    def test_whoami_json_output(self):
        """JSON output contains expected keys."""
        import json

        with self.runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                self._login_first()
                result = self.runner.invoke(
                    cli,
                    ['--config', './config.yaml', 'whoami', '-o', 'json'],
                    catch_exceptions=False,
                )

        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data['user_id'] == 'testuser'
        assert 'context' in data
        assert 'api_base' in data

    def test_whoami_not_logged_in(self):
        """Error when no PAT is configured."""
        with self.runner.isolated_filesystem():
            env = os.environ.copy()
            env.pop('CLARIFAI_PAT', None)
            with mock.patch.dict(os.environ, env, clear=True):
                result = self.runner.invoke(
                    cli,
                    ['--config', './nonexistent.yaml', 'whoami'],
                    catch_exceptions=True,
                )

        assert result.exit_code == 1
        assert 'Not logged in' in result.output or 'Not logged in' in (result.stderr or '')


class TestEnvPrefix:
    """Test cases for _env_prefix helper."""

    def test_dev_url(self):
        from clarifai.cli.base import _env_prefix

        assert _env_prefix('https://api-dev.clarifai.com') == 'dev'

    def test_staging_url(self):
        from clarifai.cli.base import _env_prefix

        assert _env_prefix('https://api-staging.clarifai.com') == 'staging'

    def test_prod_url(self):
        from clarifai.cli.base import _env_prefix

        assert _env_prefix('https://api.clarifai.com') == 'prod'
