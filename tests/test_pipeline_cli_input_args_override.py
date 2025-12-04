"""Tests for Pipeline CLI input arguments override functionality."""

import json
from unittest.mock import Mock, patch

from click.testing import CliRunner

from clarifai.cli.pipeline import run


class TestPipelineRunCommandInputArgsOverride:
    """Test cases for the pipeline run CLI command with input args override."""

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    def test_run_command_with_set_parameters(self, mock_validate_context, mock_pipeline_class):
        """Test run command with --set parameters."""
        # Mock the pipeline instance
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success', 'run_id': 'test-run-123'}
        mock_pipeline_class.return_value = mock_pipeline

        runner = CliRunner()

        # Create a proper context with current attribute like the actual Config class
        class MockContext:
            def __init__(self):
                self.pat = 'test-pat'
                self.api_base = 'https://api.clarifai.com'

            def get(self, key, default=None):
                return getattr(self, key, default)

        class MockConfig:
            def __init__(self):
                self.current = MockContext()

        ctx_obj = MockConfig()

        with runner.isolated_filesystem():
            result = runner.invoke(
                run,
                [
                    '--pipeline_id',
                    'test-pipeline',
                    '--pipeline_version_id',
                    'test-version-123',
                    '--user_id',
                    'test-user',
                    '--app_id',
                    'test-app',
                    '--nodepool_id',
                    'test-nodepool',
                    '--compute_cluster_id',
                    'test-cluster',
                    '--set',
                    'prompt=Summarize this text',
                    '--set',
                    'model=gpt-4',
                ],
                obj=ctx_obj,
            )

            assert result.exit_code == 0

            # Verify Pipeline was instantiated correctly
            mock_pipeline_class.assert_called_once()
            call_kwargs = mock_pipeline_class.call_args[1]
            assert call_kwargs['pipeline_id'] == 'test-pipeline'
            assert call_kwargs['user_id'] == 'test-user'
            assert call_kwargs['app_id'] == 'test-app'

            # Verify run was called with input_args_override
            mock_pipeline.run.assert_called_once()
            call_kwargs = mock_pipeline.run.call_args[1]
            assert 'input_args_override' in call_kwargs

            # Verify the override parameters
            input_args_override = call_kwargs['input_args_override']
            assert input_args_override is not None
            assert input_args_override.argo_args_override.parameters[0].name == 'prompt'
            assert (
                input_args_override.argo_args_override.parameters[0].value == 'Summarize this text'
            )
            assert input_args_override.argo_args_override.parameters[1].name == 'model'
            assert input_args_override.argo_args_override.parameters[1].value == 'gpt-4'

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    def test_run_command_with_overrides_file_json(
        self, mock_validate_context, mock_pipeline_class
    ):
        """Test run command with --overrides-file (JSON format)."""
        # Mock the pipeline instance
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success', 'run_id': 'test-run-456'}
        mock_pipeline_class.return_value = mock_pipeline

        runner = CliRunner()

        # Create a proper context
        class MockContext:
            def __init__(self):
                self.pat = 'test-pat'
                self.api_base = 'https://api.clarifai.com'

            def get(self, key, default=None):
                return getattr(self, key, default)

        class MockConfig:
            def __init__(self):
                self.current = MockContext()

        ctx_obj = MockConfig()

        # Create a temporary overrides file
        overrides_data = {"prompt": "Analyze this document", "temperature": "0.7"}

        with runner.isolated_filesystem():
            # Write overrides file
            with open('overrides.json', 'w') as f:
                json.dump(overrides_data, f)

            result = runner.invoke(
                run,
                [
                    '--pipeline_id',
                    'test-pipeline',
                    '--pipeline_version_id',
                    'test-version-123',
                    '--user_id',
                    'test-user',
                    '--app_id',
                    'test-app',
                    '--nodepool_id',
                    'test-nodepool',
                    '--compute_cluster_id',
                    'test-cluster',
                    '--overrides-file',
                    'overrides.json',
                ],
                obj=ctx_obj,
            )

            assert result.exit_code == 0

            # Verify Pipeline was instantiated correctly
            mock_pipeline_class.assert_called_once()

            # Verify run was called with input_args_override
            mock_pipeline.run.assert_called_once()
            call_kwargs = mock_pipeline.run.call_args[1]
            assert 'input_args_override' in call_kwargs

            # Verify the override parameters
            input_args_override = call_kwargs['input_args_override']
            assert input_args_override is not None

            # Check parameters (order might vary)
            param_map = {
                param.name: param.value
                for param in input_args_override.argo_args_override.parameters
            }
            assert param_map['prompt'] == 'Analyze this document'
            assert param_map['temperature'] == '0.7'

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    @patch('clarifai.utils.cli.from_yaml')
    def test_run_command_with_overrides_file_yaml(
        self, mock_from_yaml, mock_validate_context, mock_pipeline_class
    ):
        """Test run command with --overrides-file (YAML format)."""
        # Mock the pipeline instance
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success', 'run_id': 'test-run-789'}
        mock_pipeline_class.return_value = mock_pipeline

        # Mock YAML loading
        mock_from_yaml.return_value = {
            "system_prompt": "You are a helpful assistant",
            "max_tokens": "150",
        }

        runner = CliRunner()

        # Create a proper context
        class MockContext:
            def __init__(self):
                self.pat = 'test-pat'
                self.api_base = 'https://api.clarifai.com'

            def get(self, key, default=None):
                return getattr(self, key, default)

        class MockConfig:
            def __init__(self):
                self.current = MockContext()

        ctx_obj = MockConfig()

        with runner.isolated_filesystem():
            # Create a dummy YAML file
            with open('overrides.yaml', 'w') as f:
                f.write('system_prompt: "You are a helpful assistant"\\nmax_tokens: "150"')

            result = runner.invoke(
                run,
                [
                    '--pipeline_id',
                    'test-pipeline',
                    '--pipeline_version_id',
                    'test-version-123',
                    '--user_id',
                    'test-user',
                    '--app_id',
                    'test-app',
                    '--nodepool_id',
                    'test-nodepool',
                    '--compute_cluster_id',
                    'test-cluster',
                    '--overrides-file',
                    'overrides.yaml',
                ],
                obj=ctx_obj,
            )

            assert result.exit_code == 0

            # Verify from_yaml was called
            mock_from_yaml.assert_called_once_with('overrides.yaml')

            # Verify Pipeline was instantiated correctly
            mock_pipeline_class.assert_called_once()

            # Verify run was called with input_args_override
            mock_pipeline.run.assert_called_once()
            call_kwargs = mock_pipeline.run.call_args[1]

            # Verify the override parameters
            input_args_override = call_kwargs['input_args_override']
            assert input_args_override is not None

            # Check parameters
            param_map = {
                param.name: param.value
                for param in input_args_override.argo_args_override.parameters
            }
            assert param_map['system_prompt'] == 'You are a helpful assistant'
            assert param_map['max_tokens'] == '150'

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    def test_run_command_set_overrides_precedence(
        self, mock_validate_context, mock_pipeline_class
    ):
        """Test that --set parameters take precedence over --overrides-file."""
        # Mock the pipeline instance
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success', 'run_id': 'test-run-abc'}
        mock_pipeline_class.return_value = mock_pipeline

        runner = CliRunner()

        # Create a proper context
        class MockContext:
            def __init__(self):
                self.pat = 'test-pat'
                self.api_base = 'https://api.clarifai.com'

            def get(self, key, default=None):
                return getattr(self, key, default)

        class MockConfig:
            def __init__(self):
                self.current = MockContext()

        ctx_obj = MockConfig()

        # Create file with conflicting parameter
        overrides_data = {"prompt": "File prompt", "model": "file-model"}

        with runner.isolated_filesystem():
            # Write overrides file
            with open('overrides.json', 'w') as f:
                json.dump(overrides_data, f)

            result = runner.invoke(
                run,
                [
                    '--pipeline_id',
                    'test-pipeline',
                    '--pipeline_version_id',
                    'test-version-123',
                    '--user_id',
                    'test-user',
                    '--app_id',
                    'test-app',
                    '--nodepool_id',
                    'test-nodepool',
                    '--compute_cluster_id',
                    'test-cluster',
                    '--overrides-file',
                    'overrides.json',
                    '--set',
                    'prompt=CLI prompt override',  # Should override file value
                ],
                obj=ctx_obj,
            )

            assert result.exit_code == 0

            # Verify run was called with input_args_override
            mock_pipeline.run.assert_called_once()
            call_kwargs = mock_pipeline.run.call_args[1]

            # Verify the override parameters
            input_args_override = call_kwargs['input_args_override']
            param_map = {
                param.name: param.value
                for param in input_args_override.argo_args_override.parameters
            }

            # CLI --set should override file value
            assert param_map['prompt'] == 'CLI prompt override'
            # File value should be preserved for non-conflicting param
            assert param_map['model'] == 'file-model'

    def test_run_command_invalid_set_format(self):
        """Test run command with invalid --set format."""
        runner = CliRunner()

        # Create a proper context
        class MockContext:
            def __init__(self):
                self.pat = 'test-pat'
                self.api_base = 'https://api.clarifai.com'

            def get(self, key, default=None):
                return getattr(self, key, default)

        class MockConfig:
            def __init__(self):
                self.current = MockContext()

        ctx_obj = MockConfig()

        result = runner.invoke(
            run,
            [
                '--pipeline_id',
                'test-pipeline',
                '--pipeline_version_id',
                'test-version-123',
                '--user_id',
                'test-user',
                '--app_id',
                'test-app',
                '--nodepool_id',
                'test-nodepool',
                '--compute_cluster_id',
                'test-cluster',
                '--set',
                'invalid_format_no_equals',  # Missing =
            ],
            obj=ctx_obj,
        )

        assert result.exit_code != 0
        assert "Invalid --set format" in str(result.exception)

    def test_run_command_invalid_overrides_file(self):
        """Test run command with invalid --overrides-file."""
        runner = CliRunner()

        # Create a proper context
        class MockContext:
            def __init__(self):
                self.pat = 'test-pat'
                self.api_base = 'https://api.clarifai.com'

            def get(self, key, default=None):
                return getattr(self, key, default)

        class MockConfig:
            def __init__(self):
                self.current = MockContext()

        ctx_obj = MockConfig()

        with runner.isolated_filesystem():
            # Write invalid JSON file
            with open('invalid.json', 'w') as f:
                f.write('{ invalid json }')

            result = runner.invoke(
                run,
                [
                    '--pipeline_id',
                    'test-pipeline',
                    '--pipeline_version_id',
                    'test-version-123',
                    '--user_id',
                    'test-user',
                    '--app_id',
                    'test-app',
                    '--nodepool_id',
                    'test-nodepool',
                    '--compute_cluster_id',
                    'test-cluster',
                    '--overrides-file',
                    'invalid.json',
                ],
                obj=ctx_obj,
            )

            assert result.exit_code != 0
            assert "Failed to load overrides file" in str(result.exception)

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    def test_run_command_without_overrides_succeeds(
        self, mock_validate_context, mock_pipeline_class
    ):
        """Test that run command works without any override parameters."""
        # Mock the pipeline instance
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success', 'run_id': 'test-run-xyz'}
        mock_pipeline_class.return_value = mock_pipeline

        runner = CliRunner()

        # Create a proper context
        class MockContext:
            def __init__(self):
                self.pat = 'test-pat'
                self.api_base = 'https://api.clarifai.com'

            def get(self, key, default=None):
                return getattr(self, key, default)

        class MockConfig:
            def __init__(self):
                self.current = MockContext()

        ctx_obj = MockConfig()

        result = runner.invoke(
            run,
            [
                '--pipeline_id',
                'test-pipeline',
                '--pipeline_version_id',
                'test-version-123',
                '--user_id',
                'test-user',
                '--app_id',
                'test-app',
                '--nodepool_id',
                'test-nodepool',
                '--compute_cluster_id',
                'test-cluster',
            ],
            obj=ctx_obj,
        )

        assert result.exit_code == 0

        # Verify run was called without input_args_override
        mock_pipeline.run.assert_called_once()
        call_kwargs = mock_pipeline.run.call_args[1]
        assert call_kwargs.get('input_args_override') is None
