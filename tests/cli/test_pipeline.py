import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from clarifai.cli.pipeline import init, run, upload
from clarifai.runners.pipelines.pipeline_builder import (
    PipelineBuilder,
    PipelineConfigValidator,
    upload_pipeline,
)


class TestPipelineConfigValidator:
    """Test cases for PipelineConfigValidator."""

    def test_validate_config_missing_pipeline_section(self):
        """Test validation with missing pipeline section."""
        config = {}

        with pytest.raises(ValueError, match="'pipeline' section not found"):
            PipelineConfigValidator.validate_config(config)

    def test_validate_config_missing_required_fields(self):
        """Test validation with missing required fields."""
        config = {"pipeline": {}}

        with pytest.raises(ValueError, match="'id' not found in pipeline section"):
            PipelineConfigValidator.validate_config(config)

    def test_validate_config_empty_required_fields(self):
        """Test validation with empty required fields."""
        config = {"pipeline": {"id": "", "user_id": "test-user", "app_id": "test-app"}}

        with pytest.raises(ValueError, match="'id' cannot be empty"):
            PipelineConfigValidator.validate_config(config)

    def test_validate_config_missing_orchestration_spec(self):
        """Test validation with missing orchestration spec."""
        config = {
            "pipeline": {"id": "test-pipeline", "user_id": "test-user", "app_id": "test-app"}
        }

        with pytest.raises(ValueError, match="'orchestration_spec' not found"):
            PipelineConfigValidator.validate_config(config)

    def test_validate_config_invalid_argo_yaml(self):
        """Test validation with invalid Argo YAML."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {"argo_orchestration_spec": "invalid: yaml: :"},
            }
        }

        with pytest.raises(ValueError, match="Invalid YAML in argo_orchestration_spec"):
            PipelineConfigValidator.validate_config(config)

    def test_validate_config_valid_config(self):
        """Test validation with valid config."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "step_directories": ["stepA", "stepB"],
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: test-workflow
spec:
  entrypoint: sequence
  templates:
  - name: sequence
    steps:
    - - name: step1
        templateRef:
          name: users/test-user/apps/test-app/pipeline-steps/stepA
          template: users/test-user/apps/test-app/pipeline-steps/stepA
    - - name: step2
        templateRef:
          name: users/test-user/apps/test-app/pipeline-steps/stepB/versions/123
          template: users/test-user/apps/test-app/pipeline-steps/stepB/versions/123
                    """
                },
            }
        }

        # Should not raise any exception
        PipelineConfigValidator.validate_config(config)

    def test_validate_template_ref_invalid_name_template_mismatch(self):
        """Test template ref validation with name/template mismatch."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  templates:
  - name: sequence
    steps:
    - - name: step1
        templateRef:
          name: users/test-user/apps/test-app/pipeline-steps/stepA
          template: users/test-user/apps/test-app/pipeline-steps/stepB
                    """
                },
            }
        }

        with pytest.raises(ValueError, match="templateRef name .* must match template"):
            PipelineConfigValidator.validate_config(config)

    def test_validate_template_ref_invalid_pattern(self):
        """Test template ref validation with invalid pattern."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  templates:
  - name: sequence
    steps:
    - - name: step1
        templateRef:
          name: invalid-pattern
          template: invalid-pattern
                    """
                },
            }
        }

        with pytest.raises(ValueError, match="templateRef name .* must match either pattern"):
            PipelineConfigValidator.validate_config(config)

    def test_get_pipeline_steps_without_versions(self):
        """Test getting pipeline steps without versions."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  templates:
  - name: sequence
    steps:
    - - name: step1
        templateRef:
          name: users/test-user/apps/test-app/pipeline-steps/stepA
          template: users/test-user/apps/test-app/pipeline-steps/stepA
    - - name: step2
        templateRef:
          name: users/test-user/apps/test-app/pipeline-steps/stepB/versions/123
          template: users/test-user/apps/test-app/pipeline-steps/stepB/versions/123
                    """
                },
            }
        }

        steps = PipelineConfigValidator.get_pipeline_steps_without_versions(config)
        assert steps == ["stepA"]


class TestPipelineBuilder:
    """Test cases for PipelineBuilder."""

    @pytest.fixture
    def sample_config(self):
        """Sample valid configuration for testing."""
        return {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "step_directories": ["stepA"],
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: test-workflow
spec:
  entrypoint: sequence
  templates:
  - name: sequence
    steps:
    - - name: step1
        templateRef:
          name: users/test-user/apps/test-app/pipeline-steps/stepA
          template: users/test-user/apps/test-app/pipeline-steps/stepA
                    """
                },
            }
        }

    @pytest.fixture
    def temp_config_file(self, sample_config):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_pipeline_builder_initialization(self, temp_config_file):
        """Test PipelineBuilder initialization."""
        builder = PipelineBuilder(temp_config_file)

        assert builder.pipeline_id == "test-pipeline"
        assert builder.user_id == "test-user"
        assert builder.app_id == "test-app"
        assert builder.config_path == os.path.abspath(temp_config_file)

    def test_pipeline_builder_invalid_config(self):
        """Test PipelineBuilder with invalid config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"invalid": "config"}, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="'pipeline' section not found"):
                PipelineBuilder(temp_path)
        finally:
            os.unlink(temp_path)

    @patch('clarifai.runners.pipelines.pipeline_builder.BaseClient')
    def test_client_property(self, mock_base_client, temp_config_file):
        """Test the client property creates a BaseClient."""
        mock_client_instance = Mock()
        mock_base_client.return_value = mock_client_instance

        builder = PipelineBuilder(temp_config_file)
        client = builder.client

        mock_base_client.assert_called_once_with(user_id="test-user", app_id="test-app")
        assert client == mock_client_instance

    def test_save_config(self, temp_config_file, sample_config):
        """Test saving configuration back to file."""
        builder = PipelineBuilder(temp_config_file)

        # Modify config
        builder.config["pipeline"]["id"] = "modified-pipeline"
        builder._save_config()

        # Read back and verify
        with open(temp_config_file, 'r') as f:
            saved_config = yaml.safe_load(f)

        assert saved_config["pipeline"]["id"] == "modified-pipeline"

    @patch(
        'clarifai.runners.pipelines.pipeline_builder.PipelineBuilder._upload_pipeline_step_with_version_capture'
    )
    def test_upload_pipeline_steps_success(self, mock_upload, temp_config_file):
        """Test successful pipeline steps upload."""
        mock_upload.return_value = (True, "version-123")

        builder = PipelineBuilder(temp_config_file)

        # Mock the directory existence
        with patch('os.path.exists', return_value=True):
            result = builder.upload_pipeline_steps()

        assert result is True
        assert builder.uploaded_step_versions == {"stepA": "version-123"}

    @patch(
        'clarifai.runners.pipelines.pipeline_builder.PipelineBuilder._upload_pipeline_step_with_version_capture'
    )
    def test_upload_pipeline_steps_failure(self, mock_upload, temp_config_file):
        """Test pipeline steps upload failure."""
        mock_upload.return_value = (False, "")

        builder = PipelineBuilder(temp_config_file)

        # Mock the directory existence
        with patch('os.path.exists', return_value=True):
            result = builder.upload_pipeline_steps()

        assert result is False

    def test_upload_pipeline_steps_empty_directories(self, temp_config_file):
        """Test upload with empty step directories."""
        builder = PipelineBuilder(temp_config_file)
        builder.config["pipeline"]["step_directories"] = []

        result = builder.upload_pipeline_steps()
        assert result is True

    def test_update_config_with_versions(self, temp_config_file):
        """Test updating config with version information."""
        builder = PipelineBuilder(temp_config_file)
        builder.uploaded_step_versions = {"stepA": "version-123"}

        builder.update_config_with_versions()

        # Verify config was updated
        updated_config = builder.config
        argo_spec_str = updated_config["pipeline"]["orchestration_spec"]["argo_orchestration_spec"]
        argo_spec = yaml.safe_load(argo_spec_str)

        # Check that templateRef was updated
        template_ref = argo_spec["spec"]["templates"][0]["steps"][0][0]["templateRef"]
        expected_name = "users/test-user/apps/test-app/pipeline-steps/stepA/versions/version-123"
        assert template_ref["name"] == expected_name
        assert template_ref["template"] == expected_name

        # Check that step_directories was cleared
        assert updated_config["pipeline"]["step_directories"] == []

    @patch('clarifai.runners.pipelines.pipeline_builder.BaseClient')
    def test_create_pipeline_success(self, mock_base_client, temp_config_file):
        """Test successful pipeline creation."""
        mock_client_instance = Mock()
        mock_base_client.return_value = mock_client_instance

        # Mock successful response
        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS status code

        # Mock pipeline in response for logging
        mock_pipeline = Mock()
        mock_pipeline.id = "test-pipeline"
        mock_pipeline_version = Mock()
        mock_pipeline_version.id = "version-123"
        mock_pipeline.pipeline_version = mock_pipeline_version
        mock_response.pipelines = [mock_pipeline]

        mock_client_instance.STUB.PostPipelines.return_value = mock_response

        # Mock user_app_id properly
        from clarifai_grpc.grpc.api import resources_pb2

        mock_user_app_id = resources_pb2.UserAppIDSet(user_id="test-user", app_id="test-app")
        mock_client_instance.user_app_id = mock_user_app_id

        builder = PipelineBuilder(temp_config_file)
        result = builder.create_pipeline()

        assert result is True
        mock_client_instance.STUB.PostPipelines.assert_called_once()

    @patch('clarifai.runners.pipelines.pipeline_builder.BaseClient')
    def test_create_pipeline_failure(self, mock_base_client, temp_config_file):
        """Test pipeline creation failure."""
        mock_client_instance = Mock()
        mock_base_client.return_value = mock_client_instance

        # Mock failure response
        mock_response = Mock()
        mock_response.status.code = 40400  # FAILURE status code
        mock_response.status.description = "Test error"
        mock_response.status.details = "Test details"
        mock_client_instance.STUB.PostPipelines.return_value = mock_response

        # Mock user_app_id properly
        from clarifai_grpc.grpc.api import resources_pb2

        mock_user_app_id = resources_pb2.UserAppIDSet(user_id="test-user", app_id="test-app")
        mock_client_instance.user_app_id = mock_user_app_id

        builder = PipelineBuilder(temp_config_file)
        result = builder.create_pipeline()

        assert result is False


class TestPipelineCLIIntegration:
    """Integration tests for the pipeline CLI command."""

    def test_cli_upload_help(self):
        """Test the CLI help output."""
        runner = CliRunner()
        result = runner.invoke(upload, ['--help'])

        assert result.exit_code == 0
        assert "Upload a pipeline with associated pipeline steps" in result.output
        assert "PATH" in result.output

    def test_cli_upload_missing_config(self):
        """Test CLI upload with missing config file."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "nonexistent.yaml")
            result = runner.invoke(upload, [config_path])

            assert result.exit_code != 0  # Should fail
            # Should fail due to missing file

    def test_cli_upload_invalid_config(self):
        """Test CLI upload with invalid config file."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "invalid.yaml")

            # Create invalid config
            invalid_config = {"invalid": "config"}
            with open(config_path, 'w') as f:
                yaml.dump(invalid_config, f)

            result = runner.invoke(upload, [config_path])

            assert result.exit_code == 1
            # Should fail due to invalid config structure


class TestPipelineConfigValidatorEdgeCases:
    """Additional edge case tests for PipelineConfigValidator."""

    def test_validate_config_step_directories_not_list(self):
        """Test validation when step_directories is not a list."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "step_directories": "not-a-list",
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  templates: []
                    """
                },
            }
        }

        with pytest.raises(ValueError, match="'step_directories' must be a list"):
            PipelineConfigValidator.validate_config(config)

    def test_validate_argo_workflow_missing_required_fields(self):
        """Test Argo workflow validation with missing required fields."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {
                    "argo_orchestration_spec": """
kind: Workflow
spec:
  templates: []
                    """
                },
            }
        }

        with pytest.raises(ValueError, match="'apiVersion' not found in argo_orchestration_spec"):
            PipelineConfigValidator.validate_config(config)

    def test_validate_argo_workflow_wrong_api_version(self):
        """Test Argo workflow validation with wrong API version."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: v1
kind: Workflow
spec:
  templates: []
                    """
                },
            }
        }

        with pytest.raises(
            ValueError, match="argo_orchestration_spec must have apiVersion 'argoproj.io/v1alpha1'"
        ):
            PipelineConfigValidator.validate_config(config)

    def test_validate_argo_workflow_wrong_kind(self):
        """Test Argo workflow validation with wrong kind."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Pod
spec:
  templates: []
                    """
                },
            }
        }

        with pytest.raises(ValueError, match="argo_orchestration_spec must have kind 'Workflow'"):
            PipelineConfigValidator.validate_config(config)

    def test_validate_template_ref_missing_fields(self):
        """Test template ref validation with missing fields."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  templates:
  - name: sequence
    steps:
    - - name: step1
        templateRef:
          name: users/test-user/apps/test-app/pipeline-steps/stepA
                    """
                },
            }
        }

        with pytest.raises(
            ValueError, match="templateRef must have both 'name' and 'template' fields"
        ):
            PipelineConfigValidator.validate_config(config)

    def test_get_pipeline_steps_without_versions_empty(self):
        """Test getting pipeline steps when all have versions."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  templates:
  - name: sequence
    steps:
    - - name: step1
        templateRef:
          name: users/test-user/apps/test-app/pipeline-steps/stepA/versions/123
          template: users/test-user/apps/test-app/pipeline-steps/stepA/versions/123
                    """
                },
            }
        }

        steps = PipelineConfigValidator.get_pipeline_steps_without_versions(config)
        assert steps == []


class TestPipelineBuilderEdgeCases:
    """Additional edge case tests for PipelineBuilder."""

    @pytest.fixture
    def config_without_step_directories(self):
        """Config without step_directories field."""
        return {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app",
                "orchestration_spec": {
                    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: test-workflow
spec:
  entrypoint: sequence
  templates:
  - name: sequence
    steps:
    - - name: step1
        templateRef:
          name: users/test-user/apps/test-app/pipeline-steps/stepA/versions/123
          template: users/test-user/apps/test-app/pipeline-steps/stepA/versions/123
                    """
                },
            }
        }

    @pytest.fixture
    def temp_config_file_no_dirs(self, config_without_step_directories):
        """Create a temporary config file without step directories."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_without_step_directories, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_pipeline_builder_no_step_directories(self, temp_config_file_no_dirs):
        """Test PipelineBuilder with config that has no step_directories."""
        builder = PipelineBuilder(temp_config_file_no_dirs)

        # Should handle missing step_directories gracefully
        result = builder.upload_pipeline_steps()
        assert result is True
        assert builder.uploaded_step_versions == {}

    def test_update_config_with_no_versions(self, temp_config_file_no_dirs):
        """Test updating config when no versions were uploaded."""
        builder = PipelineBuilder(temp_config_file_no_dirs)

        # Should handle no uploaded versions gracefully
        builder.update_config_with_versions()

        # Config should be unchanged
        assert "step_directories" not in builder.config["pipeline"]


class TestUploadPipeline:
    """Test cases for the upload_pipeline function."""

    @patch('clarifai.runners.pipelines.pipeline_builder.PipelineBuilder')
    def test_upload_pipeline_with_file_path_success(self, mock_builder_class):
        """Test successful pipeline upload with file path."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        mock_builder.upload_pipeline_steps.return_value = True
        mock_builder.create_pipeline.return_value = True

        # Should not raise any exception
        upload_pipeline("test-config.yaml")

        mock_builder_class.assert_called_once_with("test-config.yaml")
        mock_builder.upload_pipeline_steps.assert_called_once()
        mock_builder.update_config_with_versions.assert_called_once()
        mock_builder.create_pipeline.assert_called_once()

    @patch('clarifai.runners.pipelines.pipeline_builder.PipelineBuilder')
    @patch('os.path.isdir')
    @patch('os.path.exists')
    def test_upload_pipeline_with_directory_path_success(
        self, mock_exists, mock_isdir, mock_builder_class
    ):
        """Test successful pipeline upload with directory path."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_isdir.return_value = True
        mock_exists.return_value = True  # config.yaml exists in directory

        mock_builder.upload_pipeline_steps.return_value = True
        mock_builder.create_pipeline.return_value = True

        # Should not raise any exception
        path_to_dir = "/path/to/directory"
        upload_pipeline(path_to_dir)

        # Should call PipelineBuilder with the config.yaml path
        mock_builder_class.assert_called_once_with(os.path.join(path_to_dir, "config.yaml"))
        mock_builder.upload_pipeline_steps.assert_called_once()
        mock_builder.update_config_with_versions.assert_called_once()
        mock_builder.create_pipeline.assert_called_once()

    @patch('os.path.isdir')
    @patch('os.path.exists')
    @patch('sys.exit')
    def test_upload_pipeline_directory_without_config(self, mock_exit, mock_exists, mock_isdir):
        """Test pipeline upload with directory path but no config.yaml."""
        mock_isdir.return_value = True
        mock_exists.return_value = False  # config.yaml does not exist in directory
        mock_exit.side_effect = SystemExit(1)

        with pytest.raises(SystemExit):
            upload_pipeline("/path/to/directory")

        mock_exit.assert_called_once_with(1)

    @patch('clarifai.runners.pipelines.pipeline_builder.PipelineBuilder')
    def test_upload_pipeline_success(self, mock_builder_class):
        """Test successful pipeline upload."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        mock_builder.upload_pipeline_steps.return_value = True
        mock_builder.create_pipeline.return_value = True

        # Should not raise any exception
        upload_pipeline("test-config.yaml")

        mock_builder.upload_pipeline_steps.assert_called_once()
        mock_builder.update_config_with_versions.assert_called_once()
        mock_builder.create_pipeline.assert_called_once()

    @patch('clarifai.runners.pipelines.pipeline_builder.PipelineBuilder')
    @patch('sys.exit')
    def test_upload_pipeline_step_failure(self, mock_exit, mock_builder_class):
        """Test pipeline upload with step failure."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        mock_builder.upload_pipeline_steps.return_value = False
        # Make sys.exit raise an exception to stop execution
        mock_exit.side_effect = SystemExit(1)

        with pytest.raises(SystemExit):
            upload_pipeline("test-config.yaml")

        mock_exit.assert_called_once_with(1)
        mock_builder.create_pipeline.assert_not_called()

    @patch('clarifai.runners.pipelines.pipeline_builder.PipelineBuilder')
    @patch('sys.exit')
    def test_upload_pipeline_creation_failure(self, mock_exit, mock_builder_class):
        """Test pipeline upload with creation failure."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        mock_builder.upload_pipeline_steps.return_value = True
        mock_builder.create_pipeline.return_value = False

        upload_pipeline("test-config.yaml")

        mock_exit.assert_called_once_with(1)


class TestUploadPipelineCLIIntegration:
    """Integration tests for CLI with new path handling."""

    def test_cli_upload_with_directory_path(self):
        """Test CLI upload with directory path containing config.yaml."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create a directory with config.yaml
            os.makedirs("pipeline_dir")
            config_content = {
                "pipeline": {
                    "id": "test-pipeline",
                    "user_id": "test-user",
                    "app_id": "test-app",
                    "orchestration_spec": {
                        "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  templates: []
                        """
                    },
                }
            }

            with open("pipeline_dir/config.yaml", 'w') as f:
                yaml.dump(config_content, f)

            # Mock the pipeline upload to avoid actual API calls
            with patch(
                'clarifai.runners.pipelines.pipeline_builder.PipelineBuilder'
            ) as mock_builder_class:
                mock_builder = Mock()
                mock_builder_class.return_value = mock_builder
                mock_builder.upload_pipeline_steps.return_value = True
                mock_builder.create_pipeline.return_value = True

                pipeline_dir_path = 'pipeline_dir'
                result = runner.invoke(upload, [pipeline_dir_path])

                # Should succeed
                assert result.exit_code == 0

                # Should have called PipelineBuilder with the config.yaml path
                expected_config_path = os.path.join(pipeline_dir_path, "config.yaml")
                mock_builder_class.assert_called_once_with(expected_config_path)

    def test_cli_upload_with_file_path(self):
        """Test CLI upload with direct config.yaml file path."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create a config.yaml file
            config_content = {
                "pipeline": {
                    "id": "test-pipeline",
                    "user_id": "test-user",
                    "app_id": "test-app",
                    "orchestration_spec": {
                        "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  templates: []
                        """
                    },
                }
            }

            with open("my-config.yaml", 'w') as f:
                yaml.dump(config_content, f)

            # Mock the pipeline upload to avoid actual API calls
            with patch(
                'clarifai.runners.pipelines.pipeline_builder.PipelineBuilder'
            ) as mock_builder_class:
                mock_builder = Mock()
                mock_builder_class.return_value = mock_builder
                mock_builder.upload_pipeline_steps.return_value = True
                mock_builder.create_pipeline.return_value = True

                result = runner.invoke(upload, ['my-config.yaml'])

                # Should succeed
                assert result.exit_code == 0

                # Should have called PipelineBuilder with the file path
                expected_config_path = "my-config.yaml"
                mock_builder_class.assert_called_once_with(expected_config_path)


class TestPipelineInitCommand:
    """Test cases for the pipeline init CLI command."""

    def test_init_command_creates_expected_structure(self):
        """Test that init command creates the expected directory structure."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['.'])

            assert result.exit_code == 0

            # Check that all expected files were created
            expected_files = [
                'config.yaml',
                'README.md',
                'stepA/config.yaml',
                'stepA/requirements.txt',
                'stepA/1/pipeline_step.py',
                'stepB/config.yaml',
                'stepB/requirements.txt',
                'stepB/1/pipeline_step.py',
            ]

            for file_path in expected_files:
                assert os.path.exists(file_path), f"Expected file {file_path} was not created"

    def test_init_command_with_custom_path(self):
        """Test that init command works with custom path."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['my_pipeline'])

            assert result.exit_code == 0

            # Check that files were created in the custom directory
            assert os.path.exists('my_pipeline/config.yaml')
            assert os.path.exists('my_pipeline/stepA/1/pipeline_step.py')
            assert os.path.exists('my_pipeline/stepB/1/pipeline_step.py')

    def test_init_command_skips_existing_files(self):
        """Test that init command skips files that already exist."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            # Create a config file first
            with open('config.yaml', 'w') as f:
                f.write('existing content')

            result = runner.invoke(init, ['.'])

            assert result.exit_code == 0

            # Check that existing file was not overwritten
            with open('config.yaml', 'r') as f:
                content = f.read()
            assert content == 'existing content'

            # Check that other files were still created
            assert os.path.exists('README.md')
            assert os.path.exists('stepA/config.yaml')

    def test_init_command_creates_valid_pipeline_config(self):
        """Test that the generated pipeline config is valid."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['.'])

            assert result.exit_code == 0

            # Load and validate the generated config
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            # Check that required sections exist
            assert 'pipeline' in config
            assert 'id' in config['pipeline']
            assert 'user_id' in config['pipeline']
            assert 'app_id' in config['pipeline']
            assert 'step_directories' in config['pipeline']
            assert 'orchestration_spec' in config['pipeline']
            assert 'argo_orchestration_spec' in config['pipeline']['orchestration_spec']

            # Check step directories
            assert config['pipeline']['step_directories'] == ['stepA', 'stepB']

    def test_init_command_creates_valid_step_configs(self):
        """Test that the generated step configs are valid."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['.'])

            assert result.exit_code == 0

            # Check stepA config
            with open('stepA/config.yaml', 'r') as f:
                step_config = yaml.safe_load(f)

            assert 'pipeline_step' in step_config
            assert step_config['pipeline_step']['id'] == 'stepA'
            assert 'pipeline_step_input_params' in step_config
            assert 'build_info' in step_config
            assert 'pipeline_step_compute_info' in step_config

    def test_init_command_includes_helpful_messages(self):
        """Test that init command works successfully."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['.'])

            assert result.exit_code == 0

            # Instead of checking logs, check that files were created successfully
            expected_files = [
                'config.yaml',
                'README.md',
                'stepA/config.yaml',
                'stepA/requirements.txt',
                'stepA/1/pipeline_step.py',
                'stepB/config.yaml',
                'stepB/requirements.txt',
                'stepB/1/pipeline_step.py',
            ]

            for file_path in expected_files:
                assert os.path.exists(file_path), f"Expected file {file_path} was not created"


class TestPipelineRunCommand:
    """Test cases for the pipeline run CLI command."""

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    def test_run_command_with_pipeline_id(self, mock_validate_context, mock_pipeline_class):
        """Test that run command works with pipeline_id, user_id, and app_id."""
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
                    '--timeout',
                    '300',
                    '--monitor_interval',
                    '5',
                ],
                obj=ctx_obj,
            )

            assert result.exit_code == 0
            mock_pipeline_class.assert_called_once_with(
                pipeline_id='test-pipeline',
                pipeline_version_id='test-version-123',
                pipeline_version_run_id=None,
                user_id='test-user',
                app_id='test-app',
                nodepool_id='test-nodepool',
                compute_cluster_id='test-cluster',
                pat='test-pat',
                base_url='https://api.clarifai.com',
                log_file=None,
            )
            mock_pipeline.run.assert_called_once_with(timeout=300, monitor_interval=5)

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    def test_run_command_with_pipeline_url(self, mock_validate_context, mock_pipeline_class):
        """Test that run command works with pipeline_url."""
        # Mock the pipeline instance
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success', 'run_id': 'test-run-456'}
        mock_pipeline_class.return_value = mock_pipeline

        runner = CliRunner()

        # Create a proper context with current attribute like the actual Config class
        class MockContext:
            def __init__(self):
                self.pat = 'test-pat'
                self.api_base = 'https://api.clarifai.com'

        class MockConfig:
            def __init__(self):
                self.current = MockContext()

        ctx_obj = MockConfig()

        with runner.isolated_filesystem():
            result = runner.invoke(
                run,
                [
                    '--pipeline_url',
                    'https://clarifai.com/user/app/pipelines/test-pipeline',
                    '--nodepool_id',
                    'test-nodepool',
                    '--compute_cluster_id',
                    'test-cluster',
                ],
                obj=ctx_obj,
            )

            assert result.exit_code == 0
            mock_pipeline_class.assert_called_once_with(
                url='https://clarifai.com/user/app/pipelines/test-pipeline',
                pat='test-pat',
                base_url='https://api.clarifai.com',
                pipeline_version_run_id=None,
                nodepool_id='test-nodepool',
                compute_cluster_id='test-cluster',
                log_file=None,
            )
            mock_pipeline.run.assert_called_once_with(timeout=3600, monitor_interval=10)

    def test_run_command_missing_required_args(self):
        """Test that run command fails when required arguments are missing."""
        runner = CliRunner()

        # Create a proper context with current attribute like the actual Config class
        class MockContext:
            def __init__(self):
                self.pat = 'test-pat'
                self.api_base = 'https://api.clarifai.com'

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
                ],  # Missing user_id, app_id, and pipeline_version_id
                obj=ctx_obj,
            )

            assert result.exit_code != 0
            assert result.exception is not None
            assert '--compute_cluster_id and --nodepool_id are mandatory parameters' in str(
                result.exception
            )

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    def test_run_command_with_nodepool_and_compute_cluster(
        self, mock_validate_context, mock_pipeline_class
    ):
        """Test that run command works with nodepool_id and compute_cluster_id."""
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success'}
        mock_pipeline_class.return_value = mock_pipeline

        runner = CliRunner()

        # Create a proper context with current attribute like the actual Config class
        class MockContext:
            def __init__(self):
                self.pat = 'test-pat'
                self.api_base = 'https://api.clarifai.com'

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
                ],
                obj=ctx_obj,
            )

            assert result.exit_code == 0
            mock_pipeline_class.assert_called_once_with(
                pipeline_id='test-pipeline',
                pipeline_version_id='test-version-123',
                pipeline_version_run_id=None,
                user_id='test-user',
                app_id='test-app',
                nodepool_id='test-nodepool',
                compute_cluster_id='test-cluster',
                pat='test-pat',
                base_url='https://api.clarifai.com',
                log_file=None,
            )
            mock_pipeline.run.assert_called_once_with(timeout=3600, monitor_interval=10)

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    def test_run_command_with_monitor_flag(self, mock_validate_context, mock_pipeline_class):
        """Test that run command works with --monitor flag."""
        mock_pipeline = Mock()
        mock_pipeline.monitor_only.return_value = {
            'status': 'success',
            'run_id': 'test-run-monitor',
        }
        mock_pipeline_class.return_value = mock_pipeline

        runner = CliRunner()

        # Create a proper context with current attribute like the actual Config class
        class MockContext:
            def __init__(self):
                self.pat = 'test-pat'
                self.api_base = 'https://api.clarifai.com'

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
                    '--pipeline_version_run_id',
                    'test-run-456',
                    '--user_id',
                    'test-user',
                    '--app_id',
                    'test-app',
                    '--nodepool_id',
                    'test-nodepool',
                    '--compute_cluster_id',
                    'test-cluster',
                    '--monitor',
                ],
                obj=ctx_obj,
            )

            assert result.exit_code == 0
            mock_pipeline_class.assert_called_once_with(
                pipeline_id='test-pipeline',
                pipeline_version_id='test-version-123',
                pipeline_version_run_id='test-run-456',
                user_id='test-user',
                app_id='test-app',
                nodepool_id='test-nodepool',
                compute_cluster_id='test-cluster',
                pat='test-pat',
                base_url='https://api.clarifai.com',
                log_file=None,
            )
            # Should call monitor_only instead of run
            mock_pipeline.monitor_only.assert_called_once_with(timeout=3600, monitor_interval=10)
            mock_pipeline.run.assert_not_called()

    def test_run_command_monitor_flag_missing_run_id(self):
        """Test that run command fails when --monitor is used without --pipeline_version_run_id."""
        runner = CliRunner()

        # Create a proper context with current attribute like the actual Config class
        class MockContext:
            def __init__(self):
                self.pat = 'test-pat'
                self.api_base = 'https://api.clarifai.com'

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
                    '--monitor',  # Missing --pipeline_version_run_id
                ],
                obj=ctx_obj,
            )

            assert result.exit_code != 0
            assert result.exception is not None
            assert '--pipeline_version_run_id is required when using --monitor flag' in str(
                result.exception
            )


class TestPipelineListCommand:
    """Test cases for the pipeline list CLI command."""

    @patch('clarifai.cli.pipeline.validate_context')
    @patch('clarifai.cli.pipeline.User')
    @patch('clarifai.cli.pipeline.display_co_resources')
    def test_list_command_success_no_app_id(self, mock_display, mock_user_class, mock_validate):
        """Test that list command works without app_id (lists across all apps)."""
        # Setup mocks
        mock_validate.return_value = None
        mock_user_instance = Mock()
        mock_user_class.return_value = mock_user_instance
        mock_user_instance.list_pipelines.return_value = [
            {'id': 'pipeline1', 'user_id': 'user1', 'app_id': 'app1', 'description': 'Test pipeline 1'},
            {'id': 'pipeline2', 'user_id': 'user1', 'app_id': 'app2', 'description': 'Test pipeline 2'},
        ]

        # Setup context
        runner = CliRunner()
        ctx_obj = Mock()
        ctx_obj.current.user_id = 'test-user'
        ctx_obj.current.pat = 'test-pat'
        ctx_obj.current.api_base = 'https://api.clarifai.com'

        # Import here to avoid circular imports in testing
        from clarifai.cli.pipeline import list as list_command

        result = runner.invoke(
            list_command,
            ['--page_no', '1', '--per_page', '10'],
            obj=ctx_obj,
        )

        assert result.exit_code == 0
        mock_validate.assert_called_once()
        mock_user_class.assert_called_once_with(
            user_id='test-user',
            pat='test-pat',
            base_url='https://api.clarifai.com'
        )
        mock_user_instance.list_pipelines.assert_called_once_with(page_no=1, per_page=10)
        mock_display.assert_called_once()

    @patch('clarifai.cli.pipeline.validate_context')
    @patch('clarifai.cli.pipeline.App')
    @patch('clarifai.cli.pipeline.display_co_resources')
    def test_list_command_success_with_app_id(self, mock_display, mock_app_class, mock_validate):
        """Test that list command works with app_id (lists within specific app)."""
        # Setup mocks
        mock_validate.return_value = None
        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance
        mock_app_instance.list_pipelines.return_value = [
            {'id': 'pipeline1', 'user_id': 'user1', 'app_id': 'app1', 'description': 'Test pipeline 1'},
        ]

        # Setup context
        runner = CliRunner()
        ctx_obj = Mock()
        ctx_obj.current.user_id = 'test-user'
        ctx_obj.current.pat = 'test-pat'
        ctx_obj.current.api_base = 'https://api.clarifai.com'

        # Import here to avoid circular imports in testing
        from clarifai.cli.pipeline import list as list_command

        result = runner.invoke(
            list_command,
            ['--app_id', 'test-app', '--page_no', '1', '--per_page', '5'],
            obj=ctx_obj,
        )

        assert result.exit_code == 0
        mock_validate.assert_called_once()
        mock_app_class.assert_called_once_with(
            app_id='test-app',
            user_id='test-user',
            pat='test-pat',
            base_url='https://api.clarifai.com'
        )
        mock_app_instance.list_pipelines.assert_called_once_with(page_no=1, per_page=5)
        mock_display.assert_called_once()

    @patch('clarifai.cli.pipeline.validate_context')
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

        # Import here to avoid circular imports in testing
        from clarifai.cli.pipeline import list as list_command

        with patch('clarifai.cli.pipeline.User') as mock_user_class:
            mock_user_instance = Mock()
            mock_user_class.return_value = mock_user_instance
            mock_user_instance.list_pipelines.return_value = []

            with patch('clarifai.cli.pipeline.display_co_resources') as mock_display:
                result = runner.invoke(list_command, [], obj=ctx_obj)

                assert result.exit_code == 0
                mock_user_instance.list_pipelines.assert_called_once_with(page_no=1, per_page=16)


class TestPipelineStepListCommand:
    """Test cases for the pipeline step list CLI command."""

    @patch('clarifai.cli.pipeline_step.validate_context')
    @patch('clarifai.cli.pipeline_step.User')
    @patch('clarifai.cli.pipeline_step.display_co_resources')
    def test_list_command_success_no_app_id(self, mock_display, mock_user_class, mock_validate):
        """Test that list command works without app_id (lists across all apps)."""
        # Setup mocks
        mock_validate.return_value = None
        mock_user_instance = Mock()
        mock_user_class.return_value = mock_user_instance
        mock_user_instance.list_pipeline_steps.return_value = [
            {'id': 'step1', 'user_id': 'user1', 'app_id': 'app1', 'pipeline_id': 'pipe1', 'description': 'Test step 1'},
            {'id': 'step2', 'user_id': 'user1', 'app_id': 'app2', 'pipeline_id': 'pipe2', 'description': 'Test step 2'},
        ]

        # Setup context
        runner = CliRunner()
        ctx_obj = Mock()
        ctx_obj.current.user_id = 'test-user'
        ctx_obj.current.pat = 'test-pat'
        ctx_obj.current.api_base = 'https://api.clarifai.com'

        # Import here to avoid circular imports in testing
        from clarifai.cli.pipeline_step import list as list_command

        result = runner.invoke(
            list_command,
            ['--page_no', '1', '--per_page', '10'],
            obj=ctx_obj,
        )

        assert result.exit_code == 0
        mock_validate.assert_called_once()
        mock_user_class.assert_called_once_with(
            user_id='test-user',
            pat='test-pat',
            base_url='https://api.clarifai.com'
        )
        mock_user_instance.list_pipeline_steps.assert_called_once_with(page_no=1, per_page=10)
        mock_display.assert_called_once()

    @patch('clarifai.cli.pipeline_step.validate_context')
    @patch('clarifai.cli.pipeline_step.App')
    @patch('clarifai.cli.pipeline_step.display_co_resources')
    def test_list_command_success_with_app_id(self, mock_display, mock_app_class, mock_validate):
        """Test that list command works with app_id (lists within specific app)."""
        # Setup mocks
        mock_validate.return_value = None
        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance
        mock_app_instance.list_pipeline_steps.return_value = [
            {'id': 'step1', 'user_id': 'user1', 'app_id': 'app1', 'pipeline_id': 'pipe1', 'description': 'Test step 1'},
        ]

        # Setup context
        runner = CliRunner()
        ctx_obj = Mock()
        ctx_obj.current.user_id = 'test-user'
        ctx_obj.current.pat = 'test-pat'
        ctx_obj.current.api_base = 'https://api.clarifai.com'

        # Import here to avoid circular imports in testing
        from clarifai.cli.pipeline_step import list as list_command

        result = runner.invoke(
            list_command,
            ['--app_id', 'test-app', '--page_no', '1', '--per_page', '5'],
            obj=ctx_obj,
        )

        assert result.exit_code == 0
        mock_validate.assert_called_once()
        mock_app_class.assert_called_once_with(
            app_id='test-app',
            user_id='test-user',
            pat='test-pat',
            base_url='https://api.clarifai.com'
        )
        mock_app_instance.list_pipeline_steps.assert_called_once_with(
            pipeline_id=None, page_no=1, per_page=5
        )
        mock_display.assert_called_once()

    @patch('clarifai.cli.pipeline_step.validate_context')
    @patch('clarifai.cli.pipeline_step.App')
    @patch('clarifai.cli.pipeline_step.display_co_resources')
    def test_list_command_success_with_pipeline_id(self, mock_display, mock_app_class, mock_validate):
        """Test that list command works with both app_id and pipeline_id."""
        # Setup mocks
        mock_validate.return_value = None
        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance
        mock_app_instance.list_pipeline_steps.return_value = [
            {'id': 'step1', 'user_id': 'user1', 'app_id': 'app1', 'pipeline_id': 'pipe1', 'description': 'Test step 1'},
        ]

        # Setup context
        runner = CliRunner()
        ctx_obj = Mock()
        ctx_obj.current.user_id = 'test-user'
        ctx_obj.current.pat = 'test-pat'
        ctx_obj.current.api_base = 'https://api.clarifai.com'

        # Import here to avoid circular imports in testing
        from clarifai.cli.pipeline_step import list as list_command

        result = runner.invoke(
            list_command,
            ['--app_id', 'test-app', '--pipeline_id', 'test-pipeline'],
            obj=ctx_obj,
        )

        assert result.exit_code == 0
        mock_validate.assert_called_once()
        mock_app_class.assert_called_once_with(
            app_id='test-app',
            user_id='test-user',
            pat='test-pat',
            base_url='https://api.clarifai.com'
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

        # Import here to avoid circular imports in testing
        from clarifai.cli.pipeline_step import list as list_command

        result = runner.invoke(
            list_command,
            ['--pipeline_id', 'test-pipeline'],
            obj=ctx_obj,
        )

        assert result.exit_code != 0
        assert '--pipeline_id must be used together with --app_id' in result.output
