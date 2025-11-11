"""Tests for Pipeline input arguments override functionality."""

import json
from unittest.mock import Mock, patch

import pytest
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.pipeline import Pipeline
from clarifai.errors import UserError


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

        # Create original orchestration spec with parameters
        original_argo_spec = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Workflow",
            "spec": {
                "arguments": {
                    "parameters": [
                        {"name": "prompt", "value": "Default prompt"},
                        {"name": "model", "value": "gpt-4"}
                    ]
                }
            }
        }

        # Mock GetPipelineVersion response
        mock_version_response = Mock()
        mock_version_response.status.code = status_code_pb2.StatusCode.SUCCESS
        mock_version_response.pipeline_version.orchestration_spec.argo_orchestration_spec.spec_json = json.dumps(original_argo_spec)
        pipeline.STUB.GetPipelineVersion.return_value = mock_version_response

        # Create input args override
        input_args_override = resources_pb2.OrchestrationArgsOverride(
            argo_args_override=resources_pb2.ArgoArgsOverride(
                parameters=[
                    resources_pb2.ArgoParameterOverride(
                        name="prompt",
                        value="Summarize this research paper"
                    )
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

        # Verify GetPipelineVersion was called
        pipeline.STUB.GetPipelineVersion.assert_called_once()

        # Verify PostPipelineVersionRuns was called
        pipeline.STUB.PostPipelineVersionRuns.assert_called_once()

        # Check that the request includes both orchestration_spec and input_args_override
        call_args = pipeline.STUB.PostPipelineVersionRuns.call_args[0][0]
        pipeline_version_run = call_args.pipeline_version_runs[0]

        # Verify orchestration_spec has merged parameters
        merged_spec = json.loads(pipeline_version_run.orchestration_spec.argo_orchestration_spec.spec_json)
        merged_params = merged_spec["spec"]["arguments"]["parameters"]

        # Check that prompt was updated and model was preserved
        param_map = {param["name"]: param["value"] for param in merged_params}
        assert param_map["prompt"] == "Summarize this research paper"
        assert param_map["model"] == "gpt-4"

        # Verify input_args_override is also set
        assert pipeline_version_run.input_args_override.argo_args_override.parameters[0].name == "prompt"
        assert pipeline_version_run.input_args_override.argo_args_override.parameters[0].value == "Summarize this research paper"

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    def test_run_with_unknown_parameter_override_fails(self, mock_init):
        """Test that overriding unknown parameters fails validation."""
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

        # Create original orchestration spec with limited parameters
        original_argo_spec = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Workflow",
            "spec": {
                "arguments": {
                    "parameters": [
                        {"name": "prompt", "value": "Default prompt"}
                    ]
                }
            }
        }

        # Mock GetPipelineVersion response
        mock_version_response = Mock()
        mock_version_response.status.code = status_code_pb2.StatusCode.SUCCESS
        mock_version_response.pipeline_version.orchestration_spec.argo_orchestration_spec.spec_json = json.dumps(original_argo_spec)
        pipeline.STUB.GetPipelineVersion.return_value = mock_version_response

        # Create input args override with unknown parameter
        input_args_override = resources_pb2.OrchestrationArgsOverride(
            argo_args_override=resources_pb2.ArgoArgsOverride(
                parameters=[
                    resources_pb2.ArgoParameterOverride(
                        name="unknown_param",
                        value="some_value"
                    )
                ]
            )
        )

        # Execute run and expect validation failure
        with pytest.raises(UserError, match="Unknown parameter 'unknown_param'. Parameter must exist in the PipelineVersion orchestration spec."):
            pipeline.run(input_args_override=input_args_override)

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    def test_run_without_input_args_override_uses_defaults(self, mock_init):
        """Test that running without input args override uses default values."""
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

        # Create original orchestration spec with parameters
        original_argo_spec = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Workflow",
            "spec": {
                "arguments": {
                    "parameters": [
                        {"name": "prompt", "value": "Default prompt"},
                        {"name": "model", "value": "gpt-4"}
                    ]
                }
            }
        }

        # Mock GetPipelineVersion response
        mock_version_response = Mock()
        mock_version_response.status.code = status_code_pb2.StatusCode.SUCCESS
        mock_version_response.pipeline_version.orchestration_spec.argo_orchestration_spec.spec_json = json.dumps(original_argo_spec)
        pipeline.STUB.GetPipelineVersion.return_value = mock_version_response

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

        # Execute run without input args override
        result = pipeline.run()

        # Verify the result
        assert result == expected_result

        # Check that the request uses original orchestration_spec
        call_args = pipeline.STUB.PostPipelineVersionRuns.call_args[0][0]
        pipeline_version_run = call_args.pipeline_version_runs[0]

        # Verify orchestration_spec has original parameters
        merged_spec = json.loads(pipeline_version_run.orchestration_spec.argo_orchestration_spec.spec_json)
        merged_params = merged_spec["spec"]["arguments"]["parameters"]

        # Check that original values are preserved
        param_map = {param["name"]: param["value"] for param in merged_params}
        assert param_map["prompt"] == "Default prompt"
        assert param_map["model"] == "gpt-4"

    def test_merge_argo_parameters_validation(self):
        """Test parameter merging validation logic directly."""
        # Create a pipeline instance
        pipeline = Pipeline(
            pipeline_id='test-pipeline',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat'
        )

        # Create orchestration spec with existing parameters
        orchestration_spec = resources_pb2.OrchestrationSpec()
        original_argo_spec = {
            "spec": {
                "arguments": {
                    "parameters": [
                        {"name": "existing_param", "value": "default_value"},
                        {"name": "another_param", "value": "another_default"}
                    ]
                }
            }
        }
        orchestration_spec.argo_orchestration_spec.spec_json = json.dumps(original_argo_spec)

        # Test 1: Valid parameter override
        valid_override_params = [
            resources_pb2.ArgoParameterOverride(
                name="existing_param",
                value="new_value"
            )
        ]

        # Should not raise exception
        pipeline._merge_argo_parameters_into_orchestration_spec(
            orchestration_spec, 
            valid_override_params
        )

        # Verify parameter was updated
        updated_spec = json.loads(orchestration_spec.argo_orchestration_spec.spec_json)
        param_map = {param["name"]: param["value"] for param in updated_spec["spec"]["arguments"]["parameters"]}
        assert param_map["existing_param"] == "new_value"
        assert param_map["another_param"] == "another_default"  # Unchanged

        # Test 2: Invalid parameter override (unknown parameter)
        invalid_override_params = [
            resources_pb2.ArgoParameterOverride(
                name="unknown_param",
                value="some_value"
            )
        ]

        with pytest.raises(UserError, match="Unknown parameter 'unknown_param'"):
            pipeline._merge_argo_parameters_into_orchestration_spec(
                orchestration_spec,
                invalid_override_params
            )

    def test_merge_argo_parameters_with_empty_orchestration_spec(self):
        """Test parameter merging with empty orchestration spec."""
        pipeline = Pipeline(
            pipeline_id='test-pipeline',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat'
        )

        # Create empty orchestration spec
        orchestration_spec = resources_pb2.OrchestrationSpec()

        override_params = [
            resources_pb2.ArgoParameterOverride(
                name="param1",
                value="value1"
            )
        ]

        # Should handle gracefully (log warning and return)
        pipeline._merge_argo_parameters_into_orchestration_spec(
            orchestration_spec,
            override_params
        )

        # Orchestration spec should remain unchanged since no argo_orchestration_spec field
        assert not orchestration_spec.HasField('argo_orchestration_spec')

    def test_merge_argo_parameters_creates_missing_structure(self):
        """Test parameter merging creates missing arguments structure."""
        pipeline = Pipeline(
            pipeline_id='test-pipeline',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat'
        )

        # Create orchestration spec with minimal structure
        orchestration_spec = resources_pb2.OrchestrationSpec()
        minimal_argo_spec = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Workflow"
        }
        orchestration_spec.argo_orchestration_spec.spec_json = json.dumps(minimal_argo_spec)

        override_params = [
            resources_pb2.ArgoParameterOverride(
                name="new_param",
                value="new_value"
            )
        ]

        # This should fail since the parameter doesn't exist in the original spec
        with pytest.raises(UserError, match="Unknown parameter 'new_param'"):
            pipeline._merge_argo_parameters_into_orchestration_spec(
                orchestration_spec,
                override_params
            )
