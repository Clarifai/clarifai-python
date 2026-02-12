"""Tests for the CLI logout command."""

import os
from collections import OrderedDict
from unittest import mock

import pytest
from click.testing import CliRunner

from clarifai.cli.base import cli
from clarifai.utils.config import Config, Context


@pytest.fixture(autouse=True)
def _clean_env():
    """Ensure CLARIFAI_PAT is not leaked from the host environment into tests."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop('CLARIFAI_PAT', None)
        yield


def _make_config(tmp_path, contexts=None, current_context='default'):
    """Build a Config, save it to disk, and return the config file path."""
    if contexts is None:
        contexts = OrderedDict(
            {
                'default': Context(
                    'default',
                    CLARIFAI_PAT='test_pat_12345',
                    CLARIFAI_USER_ID='test_user',
                    CLARIFAI_API_BASE='https://api.clarifai.com',
                ),
            }
        )
    config_path = str(tmp_path / 'config')
    cfg = Config(
        current_context=current_context,
        filename=config_path,
        contexts=contexts,
    )
    cfg.to_yaml(config_path)
    return config_path


def _multi_context_config(tmp_path):
    """Build a Config with two contexts, save to disk, return config_path."""
    contexts = OrderedDict(
        {
            'default': Context(
                'default',
                CLARIFAI_PAT='pat_default_123',
                CLARIFAI_USER_ID='user_default',
                CLARIFAI_API_BASE='https://api.clarifai.com',
            ),
            'staging': Context(
                'staging',
                CLARIFAI_PAT='pat_staging_456',
                CLARIFAI_USER_ID='user_staging',
                CLARIFAI_API_BASE='https://api-staging.clarifai.com',
            ),
        }
    )
    return _make_config(tmp_path, contexts=contexts, current_context='default')


def _load_config(config_path):
    """Reload the Config from disk after a CLI invocation."""
    return Config.from_yaml(filename=config_path)


class TestLogoutNonInteractive:
    """Tests for flag-based (non-interactive) logout."""

    def test_logout_current_clears_pat(self, tmp_path):
        """--current should clear PAT from the active context."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--current'])
        assert result.exit_code == 0
        assert "Logged out of context 'default'" in result.output
        cfg = _load_config(config_path)
        assert cfg.contexts['default']['env']['CLARIFAI_PAT'] == ''

    def test_logout_current_already_empty(self, tmp_path):
        """--current when PAT is already empty should say already logged out."""
        contexts = OrderedDict(
            {
                'default': Context(
                    'default',
                    CLARIFAI_PAT='',
                    CLARIFAI_USER_ID='test_user',
                    CLARIFAI_API_BASE='https://api.clarifai.com',
                ),
            }
        )
        config_path = _make_config(tmp_path, contexts=contexts)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--current'])
        assert result.exit_code == 0
        assert "Already logged out" in result.output

    def test_logout_current_delete_single_context(self, tmp_path):
        """--current --delete with only one context should clear PAT but keep context."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--current', '--delete'])
        assert result.exit_code == 0
        cfg = _load_config(config_path)
        assert 'default' in cfg.contexts  # context kept
        assert cfg.contexts['default']['env']['CLARIFAI_PAT'] == ''
        assert "only context" in result.output

    def test_logout_current_delete_single_already_empty(self, tmp_path):
        """--current --delete when PAT already empty should say already logged out."""
        contexts = OrderedDict(
            {
                'default': Context(
                    'default',
                    CLARIFAI_PAT='',
                    CLARIFAI_USER_ID='test_user',
                    CLARIFAI_API_BASE='https://api.clarifai.com',
                ),
            }
        )
        config_path = _make_config(tmp_path, contexts=contexts)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--current', '--delete'])
        assert result.exit_code == 0
        assert "Already logged out" in result.output
        assert "only context" in result.output

    def test_logout_current_delete_multi_context(self, tmp_path):
        """--current --delete with multiple contexts should delete and switch."""
        config_path = _multi_context_config(tmp_path=tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--current', '--delete'])
        assert result.exit_code == 0
        cfg = _load_config(config_path)
        assert 'default' not in cfg.contexts
        assert cfg.current_context == 'staging'
        assert "deleted" in result.output.lower()

    def test_logout_named_context(self, tmp_path):
        """--context <name> should clear PAT from the named context."""
        config_path = _multi_context_config(tmp_path=tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--context', 'staging'])
        assert result.exit_code == 0
        cfg = _load_config(config_path)
        assert cfg.contexts['staging']['env']['CLARIFAI_PAT'] == ''
        assert cfg.contexts['default']['env']['CLARIFAI_PAT'] == 'pat_default_123'  # untouched

    def test_logout_named_context_not_found(self, tmp_path):
        """--context <bad> should error with available contexts."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ['--config', config_path, 'logout', '--context', 'nonexistent']
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_logout_named_context_delete(self, tmp_path):
        """--context <name> --delete should remove the context."""
        config_path = _multi_context_config(tmp_path=tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ['--config', config_path, 'logout', '--context', 'staging', '--delete']
        )
        assert result.exit_code == 0
        cfg = _load_config(config_path)
        assert 'staging' not in cfg.contexts
        assert "deleted" in result.output.lower()

    def test_logout_named_context_delete_switches_current(self, tmp_path):
        """Deleting the current context via --context should switch current."""
        config_path = _multi_context_config(tmp_path=tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ['--config', config_path, 'logout', '--context', 'default', '--delete']
        )
        assert result.exit_code == 0
        cfg = _load_config(config_path)
        assert 'default' not in cfg.contexts
        assert cfg.current_context == 'staging'

    def test_logout_all(self, tmp_path):
        """--all should clear PATs from every context."""
        config_path = _multi_context_config(tmp_path=tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--all'])
        assert result.exit_code == 0
        cfg = _load_config(config_path)
        assert cfg.contexts['default']['env']['CLARIFAI_PAT'] == ''
        assert cfg.contexts['staging']['env']['CLARIFAI_PAT'] == ''
        assert "all contexts" in result.output.lower()

    def test_logout_all_already_empty(self, tmp_path):
        """--all when everything is already cleared should say so."""
        contexts = OrderedDict(
            {
                'default': Context(
                    'default',
                    CLARIFAI_PAT='',
                    CLARIFAI_USER_ID='user_default',
                    CLARIFAI_API_BASE='https://api.clarifai.com',
                ),
                'staging': Context(
                    'staging',
                    CLARIFAI_PAT='',
                    CLARIFAI_USER_ID='user_staging',
                    CLARIFAI_API_BASE='https://api-staging.clarifai.com',
                ),
            }
        )
        config_path = _make_config(tmp_path, contexts=contexts)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--all'])
        assert result.exit_code == 0
        assert "Already logged out" in result.output


class TestLogoutFlagValidation:
    """Tests for invalid flag combinations."""

    def test_delete_without_current_or_context(self, tmp_path):
        """--delete alone should error."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--delete'])
        assert result.exit_code != 0
        assert "--delete requires" in result.output

    def test_current_and_context_together(self, tmp_path):
        """--current and --context together should error."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ['--config', config_path, 'logout', '--current', '--context', 'default'],
        )
        assert result.exit_code != 0
        assert "Cannot use --current and --context together" in result.output

    def test_all_with_other_flags(self, tmp_path):
        """--all combined with --current, --context, or --delete should error."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        # --all + --current
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--all', '--current'])
        assert result.exit_code != 0
        assert "--all cannot be combined" in result.output
        # --all + --context
        result = runner.invoke(
            cli, ['--config', config_path, 'logout', '--all', '--context', 'default']
        )
        assert result.exit_code != 0
        assert "--all cannot be combined" in result.output
        # --all + --delete
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--all', '--delete'])
        assert result.exit_code != 0
        assert "--all cannot be combined" in result.output

    def test_not_logged_in(self, tmp_path):
        """Logout with no config and no env PAT should say already logged out."""
        config_path = str(tmp_path / 'nonexistent_config')
        runner = CliRunner()
        # No config file exists.  The cli() group handler creates a default
        # Config with CLARIFAI_PAT='' (the autouse _clean_env fixture ensures
        # the env var is unset), so --current finds an empty PAT.
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--current'])
        assert result.exit_code == 0
        assert "Already logged out" in result.output


class TestLogoutEnvVarWarning:
    """Tests that the CLARIFAI_PAT env var warning appears."""

    def test_warns_when_env_pat_set(self, tmp_path):
        """Should warn about env var after logout."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'env_pat_value'}):
            result = runner.invoke(cli, ['--config', config_path, 'logout', '--current'])
        assert result.exit_code == 0
        assert "CLARIFAI_PAT environment variable is still set" in result.output

    def test_no_warning_when_env_pat_unset(self, tmp_path):
        """Should not warn if env var is not set."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout', '--current'])
        assert result.exit_code == 0
        assert "CLARIFAI_PAT environment variable" not in result.output


class TestLogoutInteractive:
    """Tests for the interactive menu flow."""

    def test_interactive_cancel(self, tmp_path):
        """Choosing cancel should make no changes."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout'], input='5\n')
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        cfg = _load_config(config_path)
        assert cfg.contexts['default']['env']['CLARIFAI_PAT'] == 'test_pat_12345'

    def test_interactive_logout_current(self, tmp_path):
        """Choosing option 2 should clear current context PAT."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout'], input='2\n')
        assert result.exit_code == 0
        assert "Logged out of context 'default'" in result.output
        cfg = _load_config(config_path)
        assert cfg.contexts['default']['env']['CLARIFAI_PAT'] == ''

    def test_interactive_logout_all(self, tmp_path):
        """Choosing option 4 should clear all PATs."""
        config_path = _multi_context_config(tmp_path=tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout'], input='4\n')
        assert result.exit_code == 0
        cfg = _load_config(config_path)
        assert cfg.contexts['default']['env']['CLARIFAI_PAT'] == ''
        assert cfg.contexts['staging']['env']['CLARIFAI_PAT'] == ''

    def test_interactive_switch_context(self, tmp_path):
        """Choosing option 1 should switch to another context."""
        config_path = _multi_context_config(tmp_path=tmp_path)
        runner = CliRunner()
        # Choose switch (1), then pick the first (and only other) context (1)
        result = runner.invoke(cli, ['--config', config_path, 'logout'], input='1\n1\n')
        assert result.exit_code == 0
        cfg = _load_config(config_path)
        assert cfg.current_context == 'staging'
        assert "No credentials were cleared" in result.output

    def test_interactive_switch_no_other_contexts(self, tmp_path):
        """Switch with only one context should inform user."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout'], input='1\n')
        assert result.exit_code == 0
        assert "No other contexts available" in result.output

    def test_interactive_logout_delete_single(self, tmp_path):
        """Choosing option 3 with single context should clear but keep."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout'], input='3\n')
        assert result.exit_code == 0
        cfg = _load_config(config_path)
        assert 'default' in cfg.contexts
        assert cfg.contexts['default']['env']['CLARIFAI_PAT'] == ''
        assert "only context" in result.output

    def test_interactive_logout_delete_multi(self, tmp_path):
        """Choosing option 3 with multiple contexts should delete and switch."""
        config_path = _multi_context_config(tmp_path=tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'logout'], input='3\n')
        assert result.exit_code == 0
        cfg = _load_config(config_path)
        assert 'default' not in cfg.contexts
        assert cfg.current_context == 'staging'
