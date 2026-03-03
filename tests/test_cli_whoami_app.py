"""Tests for the CLI whoami and app CRUD commands."""

import os
from collections import OrderedDict
from unittest import mock

import pytest
from click.testing import CliRunner

from clarifai.cli.base import cli
from clarifai.utils.config import Config, Context


@pytest.fixture(autouse=True)
def _clean_env():
    """Ensure CLARIFAI_* variables are not leaked from the host environment into tests."""
    with mock.patch.dict(os.environ, {}, clear=False):
        for key in list(os.environ.keys()):
            if key.startswith('CLARIFAI_'):
                os.environ.pop(key, None)
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


class TestWhoami:
    """Tests for the whoami command."""

    def test_whoami_displays_context_user_id(self, tmp_path):
        """should display the context user ID."""
        config_path = _make_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', config_path, 'whoami'])

        assert result.exit_code == 0
        assert 'Context User ID: test_user' in result.output

    def test_whoami_handles_api_error(self, tmp_path):
        """should handle API errors gracefully."""
        config_path = _make_config(tmp_path)

        # Mock an API error response
        mock_response = mock.MagicMock()
        mock_response.status.code = 1  # ERROR
        mock_response.status.description = 'Authentication failed'

        with mock.patch('clarifai.client.user.User.get_user_info', return_value=mock_response):
            runner = CliRunner()
            result = runner.invoke(cli, ['--config', config_path, 'whoami'])

        assert result.exit_code == 0
        assert 'Context User ID: test_user' in result.output


class TestAppList:
    """Tests for the app list command."""

    def test_app_list_displays_apps(self, tmp_path):
        """should display list of apps."""
        config_path = _make_config(tmp_path)

        # Mock the list_apps generator to yield App objects
        mock_app1 = mock.MagicMock()
        mock_app1.id = 'app_1'
        mock_app1.user_id = 'test_user'

        mock_app2 = mock.MagicMock()
        mock_app2.id = 'app_2'
        mock_app2.user_id = 'test_user'

        with mock.patch(
            'clarifai.client.user.User.list_apps', return_value=iter([mock_app1, mock_app2])
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ['--config', config_path, 'app', 'ls'])

        assert result.exit_code == 0
        assert 'app_1' in result.output
        assert 'app_2' in result.output
        assert 'User ID' in result.output

    def test_app_list_no_apps_found(self, tmp_path):
        """should display message when no apps found."""
        config_path = _make_config(tmp_path)

        with mock.patch('clarifai.client.user.User.list_apps', return_value=iter([])):
            runner = CliRunner()
            result = runner.invoke(cli, ['--config', config_path, 'app', 'ls'])

        assert result.exit_code == 0
        assert 'No apps found' in result.output

    def test_app_list_with_user_id_option(self, tmp_path):
        """should list apps for a different user when --user_id is provided."""
        config_path = _make_config(tmp_path)

        # Mock the list_apps generator to yield App objects
        mock_app = mock.MagicMock()
        mock_app.id = 'other_user_app'
        mock_app.user_id = 'other_user'

        with mock.patch('clarifai.client.user.User.list_apps', return_value=iter([mock_app])):
            runner = CliRunner()
            result = runner.invoke(
                cli, ['--config', config_path, 'app', 'ls', '--user_id', 'other_user']
            )

        assert result.exit_code == 0
        assert 'other_user_app' in result.output
        assert 'other_user' in result.output


class TestAppCreate:
    """Tests for the app create command."""

    def test_app_create_success(self, tmp_path):
        """should create an app successfully."""
        config_path = _make_config(tmp_path)

        with mock.patch('clarifai.client.user.User.create_app') as mock_create:
            runner = CliRunner()
            result = runner.invoke(cli, ['--config', config_path, 'app', 'create', 'new_app'])

        assert result.exit_code == 0
        assert "App 'new_app' created successfully" in result.output
        mock_create.assert_called_once_with(app_id='new_app', base_workflow='Empty')

    def test_app_create_with_base_workflow(self, tmp_path):
        """should create an app with custom base workflow."""
        config_path = _make_config(tmp_path)

        with mock.patch('clarifai.client.user.User.create_app') as mock_create:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    '--config',
                    config_path,
                    'app',
                    'create',
                    'new_app',
                    '--base-workflow',
                    'Universal',
                ],
            )

        assert result.exit_code == 0
        mock_create.assert_called_once_with(app_id='new_app', base_workflow='Universal')


class TestAppDelete:
    """Tests for the app delete command."""

    def test_app_delete_success(self, tmp_path):
        """should delete an app successfully."""
        config_path = _make_config(tmp_path)

        with mock.patch('clarifai.client.user.User.delete_app') as mock_delete:
            runner = CliRunner()
            result = runner.invoke(
                cli, ['--config', config_path, 'app', 'delete', 'app_to_delete']
            )

        assert result.exit_code == 0
        assert "App 'app_to_delete' deleted successfully" in result.output
        mock_delete.assert_called_once_with('app_to_delete')
