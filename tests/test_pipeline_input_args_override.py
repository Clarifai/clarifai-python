"""Tests for Pipeline input arguments override functionality."""

from unittest.mock import Mock, patch

from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.pipeline import Pipeline


class TestPipelineInputArgsOverride:
    """Test cases for Pipeline input arguments override functionality."""

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    def test_run_with_input_args_override_success(self, mock_init):
        """Test successful pipeline run with input args override."""
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

        # Create input override parameters
        input_args_override = resources_pb2.OrchestrationArgsOverride(
            argo_args_override=resources_pb2.ArgoArgsOverride(
                parameters=[
                    resources_pb2.ArgoParameterOverride(name="prompt", value="Updated prompt"),
                    resources_pb2.ArgoParameterOverride(name="model", value="gpt-3.5-turbo"),
                ]
            )
        )

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

        # Execute run with input args override
        result = pipeline.run(input_args_override=input_args_override)

        # Verify the result
        assert result == expected_result

        # Verify PostPipelineVersionRuns was called (server handles parameter merging)
        pipeline.STUB.PostPipelineVersionRuns.assert_called_once()

        # Check that the request includes input_args_override
        call_args = pipeline.STUB.PostPipelineVersionRuns.call_args[0][0]
        pipeline_version_run = call_args.pipeline_version_runs[0]

        # Verify input_args_override is set correctly
        assert (
            pipeline_version_run.input_args_override.argo_args_override.parameters[0].name
            == "prompt"
        )
        assert (
            pipeline_version_run.input_args_override.argo_args_override.parameters[0].value
            == "Updated prompt"
        )
        assert (
            pipeline_version_run.input_args_override.argo_args_override.parameters[1].name
            == "model"
        )
        assert (
            pipeline_version_run.input_args_override.argo_args_override.parameters[1].value
            == "gpt-3.5-turbo"
        )

        # Verify that no orchestration_spec is set in the request (readonly field)
        assert not pipeline_version_run.HasField('orchestration_spec')

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    def test_run_without_input_args_override_uses_defaults(self, mock_init):
        """Test pipeline run without input args override uses server defaults."""
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
        mock_run.id = 'test-run-456'
        mock_run_response.pipeline_version_runs = [mock_run]
        pipeline.STUB.PostPipelineVersionRuns.return_value = mock_run_response

        # Mock the monitoring method
        expected_result = {"status": "success", "pipeline_version_run": mock_run}
        pipeline._monitor_pipeline_run = Mock(return_value=expected_result)

        # Execute run without input args override
        result = pipeline.run()

        # Verify the result
        assert result == expected_result

        # Verify PostPipelineVersionRuns was called
        pipeline.STUB.PostPipelineVersionRuns.assert_called_once()

        # Check that the request does not include input_args_override
        call_args = pipeline.STUB.PostPipelineVersionRuns.call_args[0][0]
        pipeline_version_run = call_args.pipeline_version_runs[0]

        # Verify no input_args_override is set
        assert not pipeline_version_run.HasField('input_args_override')

        # Verify that no orchestration_spec is set in the request (readonly field)
        assert not pipeline_version_run.HasField('orchestration_spec')

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    def test_run_with_empty_input_args_override(self, mock_init):
        """Test pipeline run with empty input args override."""
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

        # Create empty input override
        input_args_override = resources_pb2.OrchestrationArgsOverride(
            argo_args_override=resources_pb2.ArgoArgsOverride(parameters=[])
        )

        # Mock PostPipelineVersionRuns response
        mock_run_response = Mock()
        mock_run_response.status.code = status_code_pb2.StatusCode.SUCCESS
        mock_run = Mock()
        mock_run.id = 'test-run-789'
        mock_run_response.pipeline_version_runs = [mock_run]
        pipeline.STUB.PostPipelineVersionRuns.return_value = mock_run_response

        # Mock the monitoring method
        expected_result = {"status": "success", "pipeline_version_run": mock_run}
        pipeline._monitor_pipeline_run = Mock(return_value=expected_result)

        # Execute run with empty input args override
        result = pipeline.run(input_args_override=input_args_override)

        # Verify the result
        assert result == expected_result

        # Verify PostPipelineVersionRuns was called
        pipeline.STUB.PostPipelineVersionRuns.assert_called_once()

        # Check that the request includes empty input_args_override
        call_args = pipeline.STUB.PostPipelineVersionRuns.call_args[0][0]
        pipeline_version_run = call_args.pipeline_version_runs[0]

        # Verify empty input_args_override is set
        assert pipeline_version_run.input_args_override.argo_args_override.parameters == []
