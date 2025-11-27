"""Tests for pipeline step secrets functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.runners.pipelines.pipeline_builder import PipelineBuilder


class TestPipelineStepSecrets:
    """Test cases for pipeline step secrets feature."""

    def test_pipeline_builder_with_step_secrets(self):
        """Test that PipelineBuilder correctly handles step_version_secrets from config."""
        # Create a temporary config file with step secrets
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "step_directories": ["step1", "step2"],
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  entrypoint: sequence
  templates:
  - name: sequence
    steps:
      - - name: step-0
          templateRef:
            name: users/test-user/apps/test-app/pipeline_steps/step1
            template: users/test-user/apps/test-app/pipeline_steps/step1
""",
                },
                "config": {
                    "step_version_secrets": {
                        "step-0": {
                            "API_KEY": "users/test-user/secrets/my-api-key",
                            "DB_PASSWORD": "users/test-user/secrets/db-secret",
                        },
                        "step-1": {"EMAIL_TOKEN": "users/test-user/secrets/email-token"},
                    },
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            # Initialize builder
            with patch('clarifai.runners.pipelines.pipeline_builder.BaseClient'):
                builder = PipelineBuilder(config_path)

            # Verify config was loaded with step secrets in the config section
            assert "config" in builder.config["pipeline"]
            assert "step_version_secrets" in builder.config["pipeline"]["config"]
            step_secrets = builder.config["pipeline"]["config"]["step_version_secrets"]
            assert "step-0" in step_secrets
            assert "step-1" in step_secrets
            assert step_secrets["step-0"]["API_KEY"] == "users/test-user/secrets/my-api-key"
        finally:
            Path(config_path).unlink()

    def test_add_step_version_secrets_to_pipeline_version(self):
        """Test the _add_step_version_secrets helper method."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "step_directories": [],
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  entrypoint: sequence
  templates: []
"""
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch('clarifai.runners.pipelines.pipeline_builder.BaseClient'):
                builder = PipelineBuilder(config_path)

            # Create pipeline version proto
            pipeline_version = resources_pb2.PipelineVersion()

            # Define step secrets (flattened structure - no nested 'secrets' key)
            step_version_secrets = {
                "step-0": {
                    "API_KEY": "users/test-user/secrets/my-api-key",
                    "DB_PASSWORD": "users/test-user/secrets/db-secret",
                },
                "step-1": {"EMAIL_TOKEN": "users/test-user/secrets/email-token"},
            }

            # Call the helper method
            builder._add_step_version_secrets(pipeline_version, step_version_secrets)

            # Verify secrets were added
            assert "step-0" in pipeline_version.config.step_version_secrets
            assert "step-1" in pipeline_version.config.step_version_secrets

            # With new proto format using google.protobuf.Struct, secrets are directly accessible
            from google.protobuf import json_format

            step0_secrets_struct = pipeline_version.config.step_version_secrets["step-0"]
            step0_secrets = json_format.MessageToDict(step0_secrets_struct)
            assert step0_secrets["API_KEY"] == "users/test-user/secrets/my-api-key"
            assert step0_secrets["DB_PASSWORD"] == "users/test-user/secrets/db-secret"

            step1_secrets_struct = pipeline_version.config.step_version_secrets["step-1"]
            step1_secrets = json_format.MessageToDict(step1_secrets_struct)
            assert step1_secrets["EMAIL_TOKEN"] == "users/test-user/secrets/email-token"
        finally:
            Path(config_path).unlink()

    def test_lockfile_includes_step_secrets(self):
        """Test that lockfile generation includes step_version_secrets."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "step_directories": [],
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  entrypoint: sequence
  templates: []
""",
                },
                "config": {
                    "step_version_secrets": {
                        "step-0": {"API_KEY": "users/test-user/secrets/my-api-key"}
                    },
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch('clarifai.runners.pipelines.pipeline_builder.BaseClient'):
                builder = PipelineBuilder(config_path)

            # Generate lockfile data
            lockfile_data = builder.prepare_lockfile_with_step_versions()

            # Verify step secrets are in lockfile config section
            assert "config" in lockfile_data["pipeline"]
            assert "step_version_secrets" in lockfile_data["pipeline"]["config"]
            step_secrets = lockfile_data["pipeline"]["config"]["step_version_secrets"]
            assert "step-0" in step_secrets
            assert step_secrets["step-0"]["API_KEY"] == "users/test-user/secrets/my-api-key"
        finally:
            Path(config_path).unlink()

    def test_create_pipeline_with_step_secrets(self):
        """Test full pipeline creation with step secrets."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "step_directories": [],
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  entrypoint: sequence
  templates:
  - name: sequence
    steps:
      - - name: step-0
          templateRef:
            name: users/test-user/apps/test-app/pipeline_steps/step1/versions/v1
            template: users/test-user/apps/test-app/pipeline_steps/step1/versions/v1
""",
                },
                "config": {
                    "step_version_secrets": {
                        "step-0": {
                            "API_KEY": "users/test-user/secrets/my-api-key",
                            "DB_PASSWORD": "users/test-user/secrets/db-secret",
                        }
                    },
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch('clarifai.runners.pipelines.pipeline_builder.BaseClient') as mock_client:
                builder = PipelineBuilder(config_path)

                # Mock the STUB and response
                mock_stub = Mock()
                mock_response = Mock()
                mock_response.status.code = status_code_pb2.SUCCESS

                mock_pipeline = resources_pb2.Pipeline()
                mock_pipeline.id = "test-pipeline"
                mock_pipeline_version = resources_pb2.PipelineVersion()
                mock_pipeline_version.id = "test-version-123"
                mock_pipeline.pipeline_version.CopyFrom(mock_pipeline_version)

                mock_response.pipelines = [mock_pipeline]
                mock_stub.PostPipelines.return_value = mock_response

                builder.client.STUB = mock_stub
                builder.client.user_app_id = resources_pb2.UserAppIDSet(
                    user_id="test-user", app_id="test-app"
                )

                # Create the pipeline
                success, version_id = builder.create_pipeline()

                # Verify success
                assert success is True
                assert version_id == "test-version-123"

                # Verify PostPipelines was called
                mock_stub.PostPipelines.assert_called_once()

                # Verify the request includes step secrets
                call_args = mock_stub.PostPipelines.call_args
                request = call_args[0][0]
                assert len(request.pipelines) == 1
                pipeline = request.pipelines[0]
                assert pipeline.pipeline_version.HasField("config")
                assert "step-0" in pipeline.pipeline_version.config.step_version_secrets

                # With new proto format using google.protobuf.Struct, secrets are directly accessible
                from google.protobuf import json_format

                step_secrets_struct = pipeline.pipeline_version.config.step_version_secrets[
                    "step-0"
                ]
                step_secrets = json_format.MessageToDict(step_secrets_struct)
                assert step_secrets["API_KEY"] == "users/test-user/secrets/my-api-key"
                assert step_secrets["DB_PASSWORD"] == "users/test-user/secrets/db-secret"
        finally:
            Path(config_path).unlink()

    def test_empty_step_secrets_handling(self):
        """Test that empty step secrets are handled gracefully."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "step_directories": [],
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  entrypoint: sequence
  templates: []
"""
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch('clarifai.runners.pipelines.pipeline_builder.BaseClient'):
                builder = PipelineBuilder(config_path)

            # Generate lockfile data without secrets
            lockfile_data = builder.prepare_lockfile_with_step_versions()

            # Verify no config or step_version_secrets in lockfile when not provided
            # The config key should not exist if there are no secrets
            assert "config" not in lockfile_data["pipeline"]
        finally:
            Path(config_path).unlink()
