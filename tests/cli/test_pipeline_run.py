from unittest.mock import MagicMock, Mock, patch

import pytest
from clarifai_grpc.grpc.api.status import status_code_pb2
from click.testing import CliRunner

from clarifai.cli.pipeline_run import cancel, pause, resume


@pytest.fixture
def runner():
    """Create a Click CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_context():
    """Create a mock context object for CLI commands."""
    ctx = Mock()
    ctx.obj = Mock()
    ctx.obj.current = Mock()
    ctx.obj.current.pat = "test-pat"
    ctx.obj.current.api_base = "https://api.clarifai.com"
    ctx.obj.current.user_id = "test-user"
    return ctx


@pytest.fixture
def config_lock_data():
    """Sample config-lock.yaml data."""
    return {
        'pipeline': {
            'id': 'test-pipeline',
            'user_id': 'test-user',
            'app_id': 'test-app',
            'version_id': 'v1',
        }
    }


class TestPipelineRunPause:
    """Test cases for pause command."""

    @patch('clarifai.cli.pipeline_run.validate_context')
    @patch('clarifai.client.pipeline.Pipeline')
    def test_pause_with_positional_arg(
        self, mock_pipeline_class, mock_validate, runner, mock_context
    ):
        """Test pause command with positional argument."""
        # Setup mock
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.patch_pipeline_version_run.return_value = {'status': 'success'}

        # Run command
        result = runner.invoke(
            pause,
            [
                'test-run-id',
                '--user_id=test-user',
                '--app_id=test-app',
                '--pipeline_id=test-pipeline',
                '--pipeline_version_id=v1',
            ],
            obj=mock_context.obj,
        )

        # Assertions
        assert result.exit_code == 0
        assert 'has been paused' in result.output
        mock_pipeline.patch_pipeline_version_run.assert_called_once_with(
            pipeline_version_run_id='test-run-id',
            orchestration_status_code=status_code_pb2.JOB_PAUSED,
        )

    @patch('clarifai.cli.pipeline_run.validate_context')
    @patch('clarifai.client.pipeline.Pipeline')
    def test_pause_with_flag(self, mock_pipeline_class, mock_validate, runner, mock_context):
        """Test pause command with --pipeline_version_run_id flag."""
        # Setup mock
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.patch_pipeline_version_run.return_value = {'status': 'success'}

        # Run command
        result = runner.invoke(
            pause,
            [
                '--pipeline_version_run_id=test-run-id',
                '--user_id=test-user',
                '--app_id=test-app',
                '--pipeline_id=test-pipeline',
                '--pipeline_version_id=v1',
            ],
            obj=mock_context.obj,
        )

        # Assertions
        assert result.exit_code == 0
        assert 'has been paused' in result.output
        mock_pipeline.patch_pipeline_version_run.assert_called_once()

    @patch('clarifai.cli.pipeline_run.validate_context')
    @patch('clarifai.cli.pipeline_run.from_yaml')
    @patch('os.path.exists')
    @patch('clarifai.client.pipeline.Pipeline')
    def test_pause_with_config_lock(
        self,
        mock_pipeline_class,
        mock_exists,
        mock_from_yaml,
        mock_validate,
        runner,
        mock_context,
        config_lock_data,
    ):
        """Test pause command loading parameters from config-lock.yaml."""
        # Setup mocks
        mock_exists.return_value = True
        mock_from_yaml.return_value = config_lock_data
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.patch_pipeline_version_run.return_value = {'status': 'success'}

        # Run command
        result = runner.invoke(pause, ['test-run-id'], obj=mock_context.obj)

        # Assertions
        assert result.exit_code == 0
        mock_pipeline_class.assert_called_once_with(
            pipeline_id='test-pipeline',
            pipeline_version_id='v1',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat',
            base_url='https://api.clarifai.com',
        )

    @patch('clarifai.cli.pipeline_run.validate_context')
    def test_pause_without_run_id(self, mock_validate, runner, mock_context):
        """Test pause command fails without pipeline_version_run_id."""
        result = runner.invoke(
            pause, ['--user_id=test-user', '--app_id=test-app'], obj=mock_context.obj
        )

        assert result.exit_code != 0
        assert 'pipeline_version_run_id is required' in result.output

    @patch('clarifai.cli.pipeline_run.validate_context')
    @patch('os.path.exists')
    def test_pause_without_required_params(self, mock_exists, mock_validate, runner, mock_context):
        """Test pause command fails without required parameters and no config-lock.yaml."""
        mock_exists.return_value = False

        result = runner.invoke(pause, ['test-run-id'], obj=mock_context.obj)

        assert result.exit_code != 0
        assert 'Missing required parameters' in result.output


class TestPipelineRunCancel:
    """Test cases for cancel command."""

    @patch('clarifai.cli.pipeline_run.validate_context')
    @patch('clarifai.client.pipeline.Pipeline')
    def test_cancel_with_positional_arg(
        self, mock_pipeline_class, mock_validate, runner, mock_context
    ):
        """Test cancel command with positional argument."""
        # Setup mock
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.patch_pipeline_version_run.return_value = {'status': 'success'}

        # Run command
        result = runner.invoke(
            cancel,
            [
                'test-run-id',
                '--user_id=test-user',
                '--app_id=test-app',
                '--pipeline_id=test-pipeline',
                '--pipeline_version_id=v1',
            ],
            obj=mock_context.obj,
        )

        # Assertions
        assert result.exit_code == 0
        assert 'has been cancelled' in result.output
        mock_pipeline.patch_pipeline_version_run.assert_called_once_with(
            pipeline_version_run_id='test-run-id',
            orchestration_status_code=status_code_pb2.JOB_CANCELLED,
        )


class TestPipelineRunResume:
    """Test cases for resume command."""

    @patch('clarifai.cli.pipeline_run.validate_context')
    @patch('clarifai.client.pipeline.Pipeline')
    def test_resume_with_positional_arg(
        self, mock_pipeline_class, mock_validate, runner, mock_context
    ):
        """Test resume command with positional argument."""
        # Setup mock
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.patch_pipeline_version_run.return_value = {'status': 'success'}

        # Run command
        result = runner.invoke(
            resume,
            [
                'test-run-id',
                '--user_id=test-user',
                '--app_id=test-app',
                '--pipeline_id=test-pipeline',
                '--pipeline_version_id=v1',
            ],
            obj=mock_context.obj,
        )

        # Assertions
        assert result.exit_code == 0
        assert 'has been resumed' in result.output
        mock_pipeline.patch_pipeline_version_run.assert_called_once_with(
            pipeline_version_run_id='test-run-id',
            orchestration_status_code=status_code_pb2.JOB_RUNNING,
        )

    @patch('clarifai.cli.pipeline_run.validate_context')
    @patch('clarifai.cli.pipeline_run.from_yaml')
    @patch('os.path.exists')
    @patch('clarifai.client.pipeline.Pipeline')
    def test_resume_with_config_lock(
        self,
        mock_pipeline_class,
        mock_exists,
        mock_from_yaml,
        mock_validate,
        runner,
        mock_context,
        config_lock_data,
    ):
        """Test resume command loading parameters from config-lock.yaml."""
        # Setup mocks
        mock_exists.return_value = True
        mock_from_yaml.return_value = config_lock_data
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.patch_pipeline_version_run.return_value = {'status': 'success'}

        # Run command
        result = runner.invoke(resume, ['test-run-id'], obj=mock_context.obj)

        # Assertions
        assert result.exit_code == 0
        assert 'has been resumed' in result.output


class TestPipelineRunMonitor:
    """Test cases for monitor command."""

    @patch('clarifai.cli.pipeline_run.validate_context')
    @patch('clarifai.client.pipeline.Pipeline')
    def test_monitor_with_positional_arg(
        self, mock_pipeline_class, mock_validate, runner, mock_context
    ):
        """Test monitor command with positional argument."""
        # Setup mock
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.monitor_only.return_value = {'status': 'success', 'run_id': 'test-run-id'}

        from clarifai.cli.pipeline_run import monitor

        # Run command
        result = runner.invoke(
            monitor,
            [
                'test-run-id',
                '--user_id=test-user',
                '--app_id=test-app',
                '--pipeline_id=test-pipeline',
                '--pipeline_version_id=v1',
            ],
            obj=mock_context.obj,
        )

        # Assertions
        assert result.exit_code == 0
        mock_pipeline.monitor_only.assert_called_once_with(timeout=3600, monitor_interval=10)
        # Check that pipeline_version_run_id was set
        assert mock_pipeline.pipeline_version_run_id == 'test-run-id'

    @patch('clarifai.cli.pipeline_run.validate_context')
    @patch('clarifai.client.pipeline.Pipeline')
    def test_monitor_with_flag(self, mock_pipeline_class, mock_validate, runner, mock_context):
        """Test monitor command with --pipeline_version_run_id flag."""
        # Setup mock
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.monitor_only.return_value = {'status': 'success'}

        from clarifai.cli.pipeline_run import monitor

        # Run command
        result = runner.invoke(
            monitor,
            [
                '--pipeline_version_run_id=test-run-id',
                '--user_id=test-user',
                '--app_id=test-app',
                '--pipeline_id=test-pipeline',
                '--pipeline_version_id=v1',
            ],
            obj=mock_context.obj,
        )

        # Assertions
        assert result.exit_code == 0
        mock_pipeline.monitor_only.assert_called_once()

    @patch('clarifai.cli.pipeline_run.validate_context')
    @patch('clarifai.cli.pipeline_run.from_yaml')
    @patch('os.path.exists')
    @patch('clarifai.client.pipeline.Pipeline')
    def test_monitor_with_config_lock(
        self,
        mock_pipeline_class,
        mock_exists,
        mock_from_yaml,
        mock_validate,
        runner,
        mock_context,
        config_lock_data,
    ):
        """Test monitor command loading parameters from config-lock.yaml."""
        # Setup mocks
        mock_exists.return_value = True
        mock_from_yaml.return_value = config_lock_data
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.monitor_only.return_value = {'status': 'success'}

        from clarifai.cli.pipeline_run import monitor

        # Run command
        result = runner.invoke(monitor, ['test-run-id'], obj=mock_context.obj)

        # Assertions
        assert result.exit_code == 0
        mock_pipeline_class.assert_called_once_with(
            pipeline_id='test-pipeline',
            pipeline_version_id='v1',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat',
            base_url='https://api.clarifai.com',
        )

    @patch('clarifai.cli.pipeline_run.validate_context')
    @patch('clarifai.client.pipeline.Pipeline')
    def test_monitor_with_custom_timeout(
        self, mock_pipeline_class, mock_validate, runner, mock_context
    ):
        """Test monitor command with custom timeout and interval."""
        # Setup mock
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.monitor_only.return_value = {'status': 'success'}

        from clarifai.cli.pipeline_run import monitor

        # Run command
        result = runner.invoke(
            monitor,
            [
                'test-run-id',
                '--user_id=test-user',
                '--app_id=test-app',
                '--pipeline_id=test-pipeline',
                '--pipeline_version_id=v1',
                '--timeout=7200',
                '--monitor_interval=5',
            ],
            obj=mock_context.obj,
        )

        # Assertions
        assert result.exit_code == 0
        mock_pipeline.monitor_only.assert_called_once_with(timeout=7200, monitor_interval=5)

    @patch('clarifai.cli.pipeline_run.validate_context')
    def test_monitor_without_run_id(self, mock_validate, runner, mock_context):
        """Test monitor command fails without pipeline_version_run_id."""
        from clarifai.cli.pipeline_run import monitor

        result = runner.invoke(
            monitor, ['--user_id=test-user', '--app_id=test-app'], obj=mock_context.obj
        )

        assert result.exit_code != 0
        assert 'pipeline_version_run_id is required' in result.output
