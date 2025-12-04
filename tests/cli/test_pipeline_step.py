"""Tests for clarifai.cli.pipeline_step module."""

import logging
import os
import sys
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from clarifai.cli.pipeline_step import init, list, upload


class TestPipelineStepInitCommand:
    """Test cases for the pipeline step init CLI command."""

    def test_init_command_creates_expected_structure(self):
        """Test that init command creates the expected directory structure."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['.'])

            assert result.exit_code == 0

            # Check that all expected files were created
            expected_files = [
                'config.yaml',
                'requirements.txt',
                '1/pipeline_step.py',
            ]

            for file_path in expected_files:
                assert os.path.exists(file_path), f"Expected file {file_path} was not created"

    def test_init_command_with_custom_path(self):
        """Test that init command works with custom path."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            custom_path = 'my_pipeline_step'
            result = runner.invoke(init, [custom_path])

            assert result.exit_code == 0

            # Check that all expected files were created in the custom directory
            expected_files = [
                'my_pipeline_step/config.yaml',
                'my_pipeline_step/requirements.txt',
                'my_pipeline_step/1/pipeline_step.py',
            ]

            for file_path in expected_files:
                assert os.path.exists(file_path), f"Expected file {file_path} was not created"

    def test_init_command_skips_existing_files(self):
        """Test that init command skips files that already exist."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            # Create a pre-existing config.yaml file
            with open('config.yaml', 'w') as f:
                f.write('existing content')

            result = runner.invoke(init, ['.'])

            assert result.exit_code == 0

            # Check that the existing file was not overwritten
            with open('config.yaml', 'r') as f:
                content = f.read()
                assert content == 'existing content'

            # Check that other files were still created
            assert os.path.exists('requirements.txt')
            assert os.path.exists('1/pipeline_step.py')

    def test_init_command_creates_valid_config_content(self):
        """Test that the generated config.yaml has expected content."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['.'])

            assert result.exit_code == 0

            # Check that config.yaml has expected structure
            with open('config.yaml', 'r') as f:
                content = f.read()
                assert 'pipeline_step:' in content
                assert 'id:' in content
                assert 'user_id:' in content
                assert 'app_id:' in content
                assert 'pipeline_step_input_params:' in content
                assert 'build_info:' in content
                assert 'pipeline_step_compute_info:' in content

    def test_init_command_creates_valid_pipeline_step_content(self):
        """Test that the generated pipeline_step.py has expected content."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['.'])

            assert result.exit_code == 0

            # Check that pipeline_step.py has expected structure
            with open('1/pipeline_step.py', 'r') as f:
                content = f.read()
                assert 'import argparse' in content
                assert 'import clarifai' in content
                assert 'def main():' in content
                assert 'ArgumentParser' in content

    def test_init_command_includes_helpful_messages(self, caplog):
        """Test that init command outputs helpful messages."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})
        caplog.set_level(logging.INFO)

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['.'])

            assert result.exit_code == 0, result.output

        log_messages = caplog.messages
        assert any('Pipeline step initialization complete' in output for output in log_messages)
        assert any('Next steps:' in output for output in log_messages)
        assert any('TODO: please fill in' in output for output in log_messages)


class TestPipelineStepUploadCommand:
    """Test cases for the pipeline step upload CLI command."""

    @patch('clarifai.runners.pipeline_steps.pipeline_step_builder.upload_pipeline_step')
    def test_upload_command_with_default_path(self, mock_upload):
        """Test upload command with default path."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create a dummy config file to satisfy the path existence check
            os.makedirs('1', exist_ok=True)
            with open('config.yaml', 'w') as f:
                f.write('dummy config')

            result = runner.invoke(upload, [])

            assert result.exit_code == 0
            mock_upload.assert_called_once_with('.', False)

    @patch('clarifai.runners.pipeline_steps.pipeline_step_builder.upload_pipeline_step')
    def test_upload_command_with_custom_path(self, mock_upload):
        """Test upload command with custom path."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            custom_path = 'my_pipeline_step'
            os.makedirs(custom_path, exist_ok=True)
            os.makedirs(os.path.join(custom_path, '1'), exist_ok=True)
            with open(os.path.join(custom_path, 'config.yaml'), 'w') as f:
                f.write('dummy config')

            result = runner.invoke(upload, [custom_path])

            assert result.exit_code == 0
            mock_upload.assert_called_once_with(custom_path, False)

    @patch('clarifai.runners.pipeline_steps.pipeline_step_builder.upload_pipeline_step')
    def test_upload_command_with_skip_dockerfile_flag(self, mock_upload):
        """Test upload command with skip_dockerfile flag."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create a dummy config file to satisfy the path existence check
            os.makedirs('1', exist_ok=True)
            with open('config.yaml', 'w') as f:
                f.write('dummy config')

            result = runner.invoke(upload, ['--skip_dockerfile'])

            assert result.exit_code == 0
            mock_upload.assert_called_once_with('.', True)

    def test_upload_command_with_nonexistent_path(self):
        """Test upload command with nonexistent path fails."""
        runner = CliRunner()

        result = runner.invoke(upload, ['nonexistent_path'])

        assert result.exit_code != 0
        assert 'does not exist' in result.output


class TestPipelineStepListCommand:
    """Test cases for the pipeline step list CLI command."""

    @patch('clarifai.cli.pipeline_step.validate_context')
    @patch('clarifai.client.user.User')
    @patch('clarifai.cli.pipeline_step.display_co_resources')
    def test_list_command_success_no_app_id(self, mock_display, mock_user_class, mock_validate):
        """Test that list command works without app_id (lists across all apps)."""
        # Setup mocks
        mock_validate.return_value = None
        mock_user_instance = Mock()
        mock_user_class.return_value = mock_user_instance
        mock_user_instance.list_pipeline_steps.return_value = [
            {
                'pipeline_step_id': 'step1',
                'user_id': 'user1',
                'app_id': 'app1',
                'pipeline_step_version_id': 'version1',
                'description': 'Test step 1',
            },
            {
                'pipeline_step_id': 'step2',
                'user_id': 'user1',
                'app_id': 'app2',
                'pipeline_step_version_id': 'version2',
                'description': 'Test step 2',
            },
        ]

        # Setup context
        runner = CliRunner()
        ctx_obj = Mock()
        ctx_obj.current.user_id = 'test-user'
        ctx_obj.current.pat = 'test-pat'
        ctx_obj.current.api_base = 'https://api.clarifai.com'

        result = runner.invoke(
            list,
            ['--page_no', '1', '--per_page', '10'],
            obj=ctx_obj,
        )

        assert result.exit_code == 0
        mock_validate.assert_called_once()
        mock_user_class.assert_called_once_with(
            user_id='test-user', pat='test-pat', base_url='https://api.clarifai.com'
        )
        mock_user_instance.list_pipeline_steps.assert_called_once_with(page_no=1, per_page=10)
        mock_display.assert_called_once()

    @patch('clarifai.cli.pipeline_step.validate_context')
    @patch('clarifai.client.app.App')
    @patch('clarifai.cli.pipeline_step.display_co_resources')
    def test_list_command_success_with_app_id(self, mock_display, mock_app_class, mock_validate):
        """Test that list command works with app_id (lists within specific app)."""
        # Setup mocks
        mock_validate.return_value = None
        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance
        mock_app_instance.list_pipeline_steps.return_value = [
            {
                'pipeline_step_id': 'step1',
                'user_id': 'user1',
                'app_id': 'app1',
                'pipeline_step_version_id': 'version1',
                'description': 'Test step 1',
            },
        ]

        # Setup context
        runner = CliRunner()
        ctx_obj = Mock()
        ctx_obj.current.user_id = 'test-user'
        ctx_obj.current.pat = 'test-pat'
        ctx_obj.current.api_base = 'https://api.clarifai.com'

        result = runner.invoke(
            list,
            ['--app_id', 'test-app', '--page_no', '1', '--per_page', '5'],
            obj=ctx_obj,
        )

        assert result.exit_code == 0
        mock_validate.assert_called_once()
        mock_app_class.assert_called_once_with(
            app_id='test-app',
            user_id='test-user',
            pat='test-pat',
            base_url='https://api.clarifai.com',
        )
        mock_app_instance.list_pipeline_steps.assert_called_once_with(
            pipeline_id=None, page_no=1, per_page=5
        )
        mock_display.assert_called_once()

    @patch('clarifai.cli.pipeline_step.validate_context')
    @patch('clarifai.client.app.App')
    @patch('clarifai.cli.pipeline_step.display_co_resources')
    def test_list_command_success_with_pipeline_id(
        self, mock_display, mock_app_class, mock_validate
    ):
        """Test that list command works with both app_id and pipeline_id."""
        # Setup mocks
        mock_validate.return_value = None
        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance
        mock_app_instance.list_pipeline_steps.return_value = [
            {
                'pipeline_step_id': 'step1',
                'user_id': 'user1',
                'app_id': 'app1',
                'pipeline_step_version_id': 'version1',
                'description': 'Test step 1',
            },
        ]

        # Setup context
        runner = CliRunner()
        ctx_obj = Mock()
        ctx_obj.current.user_id = 'test-user'
        ctx_obj.current.pat = 'test-pat'
        ctx_obj.current.api_base = 'https://api.clarifai.com'

        result = runner.invoke(
            list,
            ['--app_id', 'test-app', '--pipeline_id', 'test-pipeline'],
            obj=ctx_obj,
        )

        assert result.exit_code == 0
        mock_validate.assert_called_once()
        mock_app_class.assert_called_once_with(
            app_id='test-app',
            user_id='test-user',
            pat='test-pat',
            base_url='https://api.clarifai.com',
        )
        mock_app_instance.list_pipeline_steps.assert_called_once_with(
            pipeline_id='test-pipeline', page_no=1, per_page=16
        )
        mock_display.assert_called_once()

    def test_list_command_pipeline_id_without_app_id_error(self):
        """Test that using pipeline_id without app_id raises an error."""
        runner = CliRunner()
        ctx_obj = Mock()
        ctx_obj.current.user_id = 'test-user'
        ctx_obj.current.pat = 'test-pat'
        ctx_obj.current.api_base = 'https://api.clarifai.com'

        result = runner.invoke(
            list,
            ['--pipeline_id', 'test-pipeline'],
            obj=ctx_obj,
        )

        assert result.exit_code != 0
        assert '--pipeline_id must be used together with --app_id' in result.output

    @patch('clarifai.cli.pipeline_step.validate_context')
    def test_list_command_default_parameters(self, mock_validate):
        """Test that list command uses default parameters correctly."""
        # Setup mocks
        mock_validate.return_value = None

        # Setup context
        runner = CliRunner()
        ctx_obj = Mock()
        ctx_obj.current.user_id = 'test-user'
        ctx_obj.current.pat = 'test-pat'
        ctx_obj.current.api_base = 'https://api.clarifai.com'

        with patch('clarifai.client.user.User') as mock_user_class:
            mock_user_instance = Mock()
            mock_user_class.return_value = mock_user_instance
            mock_user_instance.list_pipeline_steps.return_value = []

            with patch('clarifai.cli.pipeline_step.display_co_resources') as mock_display:
                result = runner.invoke(list, [], obj=ctx_obj)

                assert result.exit_code == 0
                mock_user_instance.list_pipeline_steps.assert_called_once_with(
                    page_no=1, per_page=16
                )


class TestPipelineStepCommandIntegration:
    """Integration tests for pipeline step CLI commands."""

    def test_init_and_upload_integration(self):
        """Test that init creates files that upload can process."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            # Initialize pipeline step
            init_result = runner.invoke(init, ['.'])
            assert init_result.exit_code == 0

            # Verify that the created structure would be valid for upload
            # (We don't actually call upload to avoid network dependencies)
            assert os.path.exists('config.yaml')
            assert os.path.exists('requirements.txt')
            assert os.path.exists('1/pipeline_step.py')

            # Check that config contains required fields for upload
            with open('config.yaml', 'r') as f:
                content = f.read()
                assert 'pipeline_step:' in content
                assert 'id:' in content


class TestPipelineStepCLIEdgeCases:
    """Test edge cases and error conditions for pipeline step CLI commands."""

    @pytest.mark.skipif(
        sys.platform == 'win32', reason="Test not relevant for Windows environment"
    )
    def test_init_with_invalid_path_permissions(self):
        """Test init command behavior with permission issues."""
        runner = CliRunner()

        # Try to create in a path that doesn't exist and can't be created
        # This simulates permission errors in a safe way
        with runner.isolated_filesystem():
            result = runner.invoke(init, ['/invalid/path/that/cannot/be/created'])

            # Should handle the error gracefully
            assert result.exit_code != 0

    def test_list_command_without_context(self):
        """Test list command behavior without proper context."""
        runner = CliRunner()

        result = runner.invoke(list, [])

        # Should fail without context
        assert result.exit_code != 0

    def test_upload_command_empty_directory(self):
        """Test upload command with empty directory."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            os.makedirs('empty_dir')

            result = runner.invoke(upload, ['empty_dir'])

            # Should fail due to missing required files
            assert result.exit_code != 0


class TestPipelineStepCommandHelp:
    """Test help output for pipeline step commands."""

    def test_init_command_help(self):
        """Test that init command shows helpful usage information."""
        runner = CliRunner()

        result = runner.invoke(init, ['--help'])

        assert result.exit_code == 0
        assert 'Initialize a new pipeline step directory structure' in result.output
        assert 'PIPELINE_STEP_PATH' in result.output

    def test_upload_command_help(self):
        """Test that upload command shows helpful usage information."""
        runner = CliRunner()

        result = runner.invoke(upload, ['--help'])

        assert result.exit_code == 0
        assert 'Upload a pipeline step to Clarifai' in result.output
        assert '--skip_dockerfile' in result.output

    def test_list_command_help(self):
        """Test that list command shows helpful usage information."""
        runner = CliRunner()

        result = runner.invoke(list, ['--help'])

        assert result.exit_code == 0
        assert 'List all pipeline steps' in result.output
        assert '--app_id' in result.output
        assert '--pipeline_id' in result.output
        assert '--page_no' in result.output
        assert '--per_page' in result.output
