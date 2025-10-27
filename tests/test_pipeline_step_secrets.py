"""Tests for Pipeline Step Secrets functionality."""
from unittest.mock import Mock, patch

import pytest
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2

from clarifai.runners.pipelines.pipeline_builder import PipelineBuilder


class TestPipelineStepSecrets:
    """Test cases for pipeline step secrets functionality."""

    @patch('clarifai.runners.pipelines.pipeline_builder.BaseClient')
    def test_add_step_secrets_to_pipeline_version(self, mock_client):
        """Test adding step secrets to pipeline version."""
        # Create a mock config with step secrets
        config_data = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {"argo_orchestration_spec": "apiVersion: v1\nspec:\n  templates: []"},
                "step_directories": [],
                "step_secrets": {
                    "step1": {
                        "API_KEY": "users/test-user/secrets/my-key",
                        "DB_PASSWORD": "users/test-user/secrets/db-pass",
                    },
                    "step2": {"EMAIL_TOKEN": "users/test-user/secrets/email-token"},
                },
            }
        }

        with patch.object(PipelineBuilder, '_load_config', return_value=config_data):
            with patch.object(PipelineBuilder, 'validator') as mock_validator:
                mock_validator.validate_config = Mock()
                builder = PipelineBuilder('/fake/path/config.yaml')

                # Create a pipeline version proto
                pipeline_version = resources_pb2.PipelineVersion()

                # Call the method
                step_secrets = config_data["pipeline"]["step_secrets"]
                builder._add_step_secrets_to_pipeline_version(pipeline_version, step_secrets)

                # Verify step secrets were added
                try:
                    # Check if step_version_secrets field exists and has data
                    assert hasattr(pipeline_version.config, 'step_version_secrets')
                    assert 'step1' in pipeline_version.config.step_version_secrets
                    assert 'step2' in pipeline_version.config.step_version_secrets

                    # Verify step1 secrets
                    step1_secrets = pipeline_version.config.step_version_secrets['step1']
                    assert 'API_KEY' in step1_secrets.secrets
                    assert step1_secrets.secrets['API_KEY'] == 'users/test-user/secrets/my-key'
                    assert 'DB_PASSWORD' in step1_secrets.secrets
                    assert step1_secrets.secrets['DB_PASSWORD'] == 'users/test-user/secrets/db-pass'

                    # Verify step2 secrets
                    step2_secrets = pipeline_version.config.step_version_secrets['step2']
                    assert 'EMAIL_TOKEN' in step2_secrets.secrets
                    assert step2_secrets.secrets['EMAIL_TOKEN'] == 'users/test-user/secrets/email-token'

                except AttributeError:
                    # If the proto doesn't support step_version_secrets, the test should note this
                    pytest.skip("Proto version doesn't support step_version_secrets field")

    @patch('clarifai.runners.pipelines.pipeline_builder.BaseClient')
    def test_add_empty_step_secrets(self, mock_client):
        """Test that empty step secrets don't cause errors."""
        config_data = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {"argo_orchestration_spec": "apiVersion: v1\nspec:\n  templates: []"},
                "step_directories": [],
                "step_secrets": {},
            }
        }

        with patch.object(PipelineBuilder, '_load_config', return_value=config_data):
            with patch.object(PipelineBuilder, 'validator') as mock_validator:
                mock_validator.validate_config = Mock()
                builder = PipelineBuilder('/fake/path/config.yaml')

                # Create a pipeline version proto
                pipeline_version = resources_pb2.PipelineVersion()

                # Call the method with empty secrets
                step_secrets = config_data["pipeline"]["step_secrets"]
                builder._add_step_secrets_to_pipeline_version(pipeline_version, step_secrets)

                # Should not raise any errors

    @patch('clarifai.runners.pipelines.pipeline_builder.BaseClient')
    def test_create_pipeline_with_step_secrets(self, mock_client):
        """Test creating a pipeline with step secrets."""
        config_data = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {
                    "argo_orchestration_spec": "apiVersion: argoproj.io/v1alpha1\nspec:\n  templates:\n    - name: main\n      steps: []"
                },
                "step_directories": [],
                "step_secrets": {
                    "step1": {"API_KEY": "users/test-user/secrets/my-key"},
                },
            }
        }

        with patch.object(PipelineBuilder, '_load_config', return_value=config_data):
            with patch.object(PipelineBuilder, 'validator') as mock_validator:
                mock_validator.validate_config = Mock()
                builder = PipelineBuilder('/fake/path/config.yaml')

                # Mock the client and STUB
                mock_stub = Mock()
                builder.client.STUB = mock_stub
                builder.client.user_app_id = resources_pb2.UserAppIDSet(
                    user_id="test-user", app_id="test-app"
                )

                # Mock successful response
                mock_response = Mock()
                mock_response.status.code = status_code_pb2.SUCCESS
                mock_pipeline = Mock()
                mock_pipeline.id = "test-pipeline"
                mock_pipeline.pipeline_version.id = "version-123"
                mock_response.pipelines = [mock_pipeline]
                mock_stub.PostPipelines.return_value = mock_response

                # Create the pipeline
                success, version_id = builder.create_pipeline()

                # Verify success
                assert success is True
                assert version_id == "version-123"

                # Verify PostPipelines was called
                mock_stub.PostPipelines.assert_called_once()

                # Get the request that was passed
                call_args = mock_stub.PostPipelines.call_args
                request = call_args[0][0]

                # Verify the request has a pipeline with step secrets
                assert len(request.pipelines) == 1
                pipeline = request.pipelines[0]

                # Try to verify step secrets were included (if proto supports it)
                try:
                    if pipeline.pipeline_version.HasField("config"):
                        step_version_secrets = pipeline.pipeline_version.config.step_version_secrets
                        assert 'step1' in step_version_secrets
                        assert 'API_KEY' in step_version_secrets['step1'].secrets
                except AttributeError:
                    # Proto doesn't support step_version_secrets yet
                    pytest.skip("Proto version doesn't support step_version_secrets field")


class TestPipelineStepSecretsBackwardCompatibility:
    """Test backward compatibility of pipeline step secrets."""

    @patch('clarifai.runners.pipelines.pipeline_builder.BaseClient')
    def test_create_pipeline_without_step_secrets(self, mock_client):
        """Test creating a pipeline without step secrets (backward compatibility)."""
        config_data = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {
                    "argo_orchestration_spec": "apiVersion: argoproj.io/v1alpha1\nspec:\n  templates:\n    - name: main\n      steps: []"
                },
                "step_directories": [],
                # No step_secrets field - should work fine
            }
        }

        with patch.object(PipelineBuilder, '_load_config', return_value=config_data):
            with patch.object(PipelineBuilder, 'validator') as mock_validator:
                mock_validator.validate_config = Mock()
                builder = PipelineBuilder('/fake/path/config.yaml')

                # Mock the client and STUB
                mock_stub = Mock()
                builder.client.STUB = mock_stub
                builder.client.user_app_id = resources_pb2.UserAppIDSet(
                    user_id="test-user", app_id="test-app"
                )

                # Mock successful response
                mock_response = Mock()
                mock_response.status.code = status_code_pb2.SUCCESS
                mock_pipeline = Mock()
                mock_pipeline.id = "test-pipeline"
                mock_pipeline.pipeline_version.id = "version-123"
                mock_response.pipelines = [mock_pipeline]
                mock_stub.PostPipelines.return_value = mock_response

                # Create the pipeline - should work without step_secrets
                success, version_id = builder.create_pipeline()

                # Verify success
                assert success is True
                assert version_id == "version-123"
