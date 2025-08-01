from unittest.mock import Mock, patch

import pytest
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format

from clarifai.client.pipeline import Pipeline
from clarifai.errors import UserError


class TestPipelineClient:
    """Test cases for the Pipeline client."""

    def test_pipeline_initialization_with_url(self):
        """Test that Pipeline client initializes correctly with URL."""
        url = "https://clarifai.com/test-user/test-app/pipelines/test-pipeline"

        with patch('clarifai.client.pipeline.BaseClient.__init__') as mock_init:
            mock_init.return_value = None
            pipeline = Pipeline(url=url, pat='test-pat')

            assert pipeline.pipeline_id == 'test-pipeline'
            assert pipeline.user_id == 'test-user'
            assert pipeline.app_id == 'test-app'

    def test_pipeline_initialization_with_ids(self):
        """Test that Pipeline client initializes correctly with separate IDs."""
        with patch('clarifai.client.pipeline.BaseClient.__init__') as mock_init:
            mock_init.return_value = None
            pipeline = Pipeline(
                pipeline_id='test-pipeline', user_id='test-user', app_id='test-app', pat='test-pat'
            )

            assert pipeline.pipeline_id == 'test-pipeline'
            assert pipeline.user_id == 'test-user'
            assert pipeline.app_id == 'test-app'

    def test_pipeline_initialization_validation(self):
        """Test that Pipeline client validation works correctly."""
        # Test with both url and pipeline_id
        with pytest.raises(UserError, match="You can only specify one of url or pipeline_id"):
            Pipeline(
                url="https://clarifai.com/test-user/test-app/pipelines/test-pipeline",
                pipeline_id="test-pipeline",
                pat='test-pat',
            )

        # Test with neither url nor pipeline_id
        with pytest.raises(UserError, match="You must specify one of url or pipeline_id"):
            Pipeline(pat='test-pat')

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    def test_run_success(self, mock_init):
        """Test successful pipeline run."""
        mock_init.return_value = None

        pipeline = Pipeline(
            pipeline_id='test-pipeline',
            pipeline_version_id='test-version-123',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat',
        )

        # Mock the required attributes
        pipeline.user_app_id = resources_pb2.UserAppIDSet(user_id="test-user", app_id="test-app")
        pipeline.STUB = Mock()
        pipeline.auth_helper = Mock()
        pipeline.auth_helper.metadata = []

        # Mock PostPipelineVersionRuns response
        mock_run_response = Mock()
        mock_run_response.status.code = status_code_pb2.StatusCode.SUCCESS
        mock_run = Mock()
        mock_run.id = 'test-run-123'
        mock_run_response.pipeline_version_runs = [mock_run]
        pipeline.STUB.PostPipelineVersionRuns.return_value = mock_run_response

        # Mock the monitoring method
        expected_result = {"status": "success", "pipeline_version_run": mock_run}
        pipeline._monitor_pipeline_run = Mock(return_value=expected_result)

        # Execute run
        result = pipeline.run()

        # Verify the result
        assert result == expected_result
        pipeline.STUB.PostPipelineVersionRuns.assert_called_once()
        pipeline._monitor_pipeline_run.assert_called_once_with('test-run-123', 3600, 10)

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    def test_run_failure_to_start(self, mock_init):
        """Test pipeline run failure to start."""
        mock_init.return_value = None

        pipeline = Pipeline(
            pipeline_id='test-pipeline',
            pipeline_version_id='test-version-123',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat',
        )

        # Mock the required attributes
        pipeline.user_app_id = resources_pb2.UserAppIDSet(user_id="test-user", app_id="test-app")
        pipeline.STUB = Mock()
        pipeline.auth_helper = Mock()
        pipeline.auth_helper.metadata = []

        # Mock failed PostPipelineVersionRuns response
        mock_response = Mock()
        mock_response.status.code = (
            status_code_pb2.StatusCode.MIXED_STATUS
        )  # Use a valid status code
        mock_response.status.description = "Failed to start pipeline"
        pipeline.STUB.PostPipelineVersionRuns.return_value = mock_response

        # Execute run and expect failure
        with pytest.raises(
            UserError, match="Failed to start pipeline run: Failed to start pipeline"
        ):
            pipeline.run()

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    @patch('time.time')
    @patch('time.sleep')
    def test_monitor_pipeline_run_success(self, mock_sleep, mock_time, mock_init):
        """Test successful pipeline run monitoring."""
        mock_init.return_value = None
        # Provide enough time values for the test including logging calls
        mock_time.side_effect = [0] + [10] * 20  # Start at 0, then always return 10

        pipeline = Pipeline(
            pipeline_id='test-pipeline',
            pipeline_version_id='test-version-123',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat',
        )

        # Mock the required attributes
        pipeline.user_app_id = resources_pb2.UserAppIDSet(user_id="test-user", app_id="test-app")
        pipeline.STUB = Mock()
        pipeline.auth_helper = Mock()
        pipeline.auth_helper.metadata = []

        # Mock successful GetPipelineVersionRun response
        # use real proto instead of mock for pipeline_version_run response
        get_response = service_pb2.SinglePipelineVersionRunResponse(
            status=status_pb2.Status(code=status_code_pb2.StatusCode.SUCCESS),
            pipeline_version_run=resources_pb2.PipelineVersionRun(
                id='test-run-123',
                orchestration_status=resources_pb2.OrchestrationStatus(
                    status=status_pb2.Status(code=status_code_pb2.StatusCode.SUCCESS)
                ),
            ),
        )
        pipeline.STUB.GetPipelineVersionRun.return_value = get_response

        # Mock log display
        pipeline._display_new_logs = Mock()

        # Execute monitoring
        result = pipeline._monitor_pipeline_run('test-run-123', 3600, 10)

        # Verify the result
        assert result["status"] == "success"
        assert result["pipeline_version_run"] == json_format.MessageToDict(
            get_response.pipeline_version_run, preserving_proto_field_name=True
        )
        pipeline.STUB.GetPipelineVersionRun.assert_called_once()
        pipeline._display_new_logs.assert_called_once()

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    @patch('time.time')
    @patch('time.sleep')
    def test_monitor_pipeline_run_timeout(self, mock_sleep, mock_time, mock_init):
        """Test pipeline run monitoring timeout."""
        mock_init.return_value = None
        # Provide timeout after 1 hour+ but enough values for logging
        mock_time.side_effect = [0] + [3700] * 20

        pipeline = Pipeline(
            pipeline_id='test-pipeline',
            pipeline_version_id='test-version-123',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat',
        )

        # Mock the required attributes
        pipeline.user_app_id = resources_pb2.UserAppIDSet(user_id="test-user", app_id="test-app")
        pipeline.STUB = Mock()
        pipeline.auth_helper = Mock()
        pipeline.auth_helper.metadata = []

        # Mock running GetPipelineVersionRun response
        # use real proto instead of mock for pipeline_version_run response
        get_response = service_pb2.SinglePipelineVersionRunResponse(
            status=status_pb2.Status(code=status_code_pb2.StatusCode.SUCCESS),
            pipeline_version_run=resources_pb2.PipelineVersionRun(
                id='test-run-123',
                orchestration_status=resources_pb2.OrchestrationStatus(
                    status=status_pb2.Status(code=status_code_pb2.StatusCode.MIXED_STATUS)
                ),
            ),
        )
        pipeline.STUB.GetPipelineVersionRun.return_value = get_response

        # Mock log display
        pipeline._display_new_logs = Mock()

        # Execute monitoring
        result = pipeline._monitor_pipeline_run('test-run-123', 3600, 10)

        # Verify timeout result
        assert result["status"] == "timeout"

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    def test_monitor_only_success(self, mock_init):
        """Test successful monitor_only method."""
        mock_init.return_value = None

        pipeline = Pipeline(
            pipeline_id='test-pipeline',
            pipeline_version_id='test-version-123',
            pipeline_version_run_id='test-run-456',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat',
        )

        # Mock the monitoring method
        expected_result = {"status": "success", "pipeline_version_run": Mock()}
        pipeline._monitor_pipeline_run = Mock(return_value=expected_result)

        # Execute monitor_only
        result = pipeline.monitor_only(timeout=1800, monitor_interval=5)

        # Verify the result
        assert result == expected_result
        pipeline._monitor_pipeline_run.assert_called_once_with('test-run-456', 1800, 5)

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    def test_monitor_only_missing_run_id(self, mock_init):
        """Test monitor_only method fails when pipeline_version_run_id is missing."""
        mock_init.return_value = None

        pipeline = Pipeline(
            pipeline_id='test-pipeline',
            pipeline_version_id='test-version-123',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat',
        )

        # Explicitly set pipeline_version_run_id to None to test the validation
        pipeline.pipeline_version_run_id = None

        # Execute monitor_only without pipeline_version_run_id and expect failure
        with pytest.raises(
            UserError, match="pipeline_version_run_id is required for monitoring existing runs"
        ):
            pipeline.monitor_only()
