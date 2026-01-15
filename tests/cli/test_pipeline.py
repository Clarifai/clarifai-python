import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from clarifai.cli.pipeline import init, run, upload
from clarifai.cli.pipeline_template import info, list_templates
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
          name: users/test-user/apps/test-app/pipeline_steps/stepA
          template: users/test-user/apps/test-app/pipeline_steps/stepA
    - - name: step2
        templateRef:
          name: users/test-user/apps/test-app/pipeline_steps/stepB/versions/123
          template: users/test-user/apps/test-app/pipeline_steps/stepB/versions/123
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
          name: users/test-user/apps/test-app/pipeline_steps/stepA
          template: users/test-user/apps/test-app/pipeline_steps/stepB
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
          name: users/test-user/apps/test-app/pipeline_steps/stepA
          template: users/test-user/apps/test-app/pipeline_steps/stepA
    - - name: step2
        templateRef:
          name: users/test-user/apps/test-app/pipeline_steps/stepB/versions/123
          template: users/test-user/apps/test-app/pipeline_steps/stepB/versions/123
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
          name: users/test-user/apps/test-app/pipeline_steps/stepA
          template: users/test-user/apps/test-app/pipeline_steps/stepA
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
        assert result is False

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
        success, version_id = builder.create_pipeline()

        assert success is True
        assert version_id == "version-123"
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
        success, version_id = builder.create_pipeline()

        assert success is False
        assert version_id == ""


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
          name: users/test-user/apps/test-app/pipeline_steps/stepA
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
          name: users/test-user/apps/test-app/pipeline_steps/stepA/versions/123
          template: users/test-user/apps/test-app/pipeline_steps/stepA/versions/123
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
          name: users/test-user/apps/test-app/pipeline_steps/stepA/versions/123
          template: users/test-user/apps/test-app/pipeline_steps/stepA/versions/123
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
        assert result is False
        assert builder.uploaded_step_versions == {}

    def test_update_config_with_no_versions(self, temp_config_file_no_dirs):
        """Test that config remains unchanged when no versions were uploaded."""
        builder = PipelineBuilder(temp_config_file_no_dirs)

        # Config should be unchanged when no versions are available
        assert "step_directories" not in builder.config["pipeline"]


class TestUploadPipeline:
    """Test cases for the upload_pipeline function."""

    @patch('clarifai.runners.pipelines.pipeline_builder.PipelineBuilder')
    def test_upload_pipeline_with_file_path_success(self, mock_builder_class):
        """Test successful pipeline upload with file path."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        mock_builder.upload_pipeline_steps.return_value = True
        mock_builder.create_pipeline.return_value = (True, "version-123")

        # Should not raise any exception
        upload_pipeline("test-config.yaml")

        mock_builder_class.assert_called_once_with("test-config.yaml")
        mock_builder.upload_pipeline_steps.assert_called_once()
        mock_builder.create_pipeline.assert_called_once()
        # Note: config.yaml is no longer modified during pipeline upload

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
        mock_builder.create_pipeline.return_value = (True, "version-123")

        # Should not raise any exception
        path_to_dir = "/path/to/directory"
        upload_pipeline(path_to_dir)

        # Should call PipelineBuilder with the config.yaml path
        mock_builder_class.assert_called_once_with(os.path.join(path_to_dir, "config.yaml"))
        mock_builder.upload_pipeline_steps.assert_called_once()
        mock_builder.create_pipeline.assert_called_once()
        # Note: config.yaml is no longer modified during pipeline upload

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
        mock_builder.create_pipeline.return_value = (True, "version-123")

        # Should not raise any exception
        upload_pipeline("test-config.yaml")

        mock_builder.upload_pipeline_steps.assert_called_once()
        mock_builder.create_pipeline.assert_called_once()
        # Note: config.yaml is no longer modified during pipeline upload

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
        mock_builder.create_pipeline.return_value = (False, "")

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
                mock_builder.create_pipeline.return_value = (True, "version-123")

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
                mock_builder.create_pipeline.return_value = (True, "version-123")

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
            # Provide inputs for the interactive prompts
            inputs = "test-user\ntest-app\nhello-world-pipeline\n2\nstepA\nstepB\n"
            result = runner.invoke(init, ['.'], input=inputs)

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

    def test_init_command_with_custom_inputs(self):
        """Test that init command works with custom user inputs."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            # Provide custom inputs
            inputs = "custom-user\ncustom-app\ncustom-pipeline\n3\ndata-prep\nmodel-train\nmodel-deploy\n"
            result = runner.invoke(init, ['.'], input=inputs)

            assert result.exit_code == 0

            # Check that custom step directories were created
            assert os.path.exists('data-prep/config.yaml')
            assert os.path.exists('model-train/config.yaml')
            assert os.path.exists('model-deploy/config.yaml')

            # Verify the pipeline config contains the custom values
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            assert config['pipeline']['id'] == 'custom-pipeline'
            assert config['pipeline']['user_id'] == 'custom-user'
            assert config['pipeline']['app_id'] == 'custom-app'
            assert config['pipeline']['step_directories'] == [
                'data-prep',
                'model-train',
                'model-deploy',
            ]

    def test_init_command_with_custom_path(self):
        """Test that init command works with custom path."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            inputs = "test-user\ntest-app\nhello-world-pipeline\n2\nstepA\nstepB\n"
            result = runner.invoke(init, ['my_pipeline'], input=inputs)

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

            inputs = "test-user\ntest-app\nhello-world-pipeline\n2\nstepA\nstepB\n"
            result = runner.invoke(init, ['.'], input=inputs)

            assert result.exit_code == 0

            # Check that existing file was not overwritten
            with open('config.yaml', 'r') as f:
                content = f.read()
            assert content == 'existing content'

            # Check that other files were still created
            assert os.path.exists('README.md')
            assert os.path.exists('stepA/config.yaml')

    def test_init_command_creates_valid_pipeline_config(self):
        """Test that the generated pipeline config is valid and has no TODO comments."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            inputs = "test-user\ntest-app\ntest-pipeline\n2\nstepA\nstepB\n"
            result = runner.invoke(init, ['.'], input=inputs)

            assert result.exit_code == 0

            # Load and validate the generated config
            with open('config.yaml', 'r') as f:
                config_content = f.read()
                config = yaml.safe_load(config_content)

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

            # Check that actual values are used, not placeholders
            assert config['pipeline']['id'] == 'test-pipeline'
            assert config['pipeline']['user_id'] == 'test-user'
            assert config['pipeline']['app_id'] == 'test-app'

            # Ensure no TODO comments exist
            assert 'TODO' not in config_content

    def test_init_command_creates_valid_step_configs(self):
        """Test that the generated step configs are valid and have no TODO comments."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            inputs = "test-user\ntest-app\ntest-pipeline\n2\nstepA\nstepB\n"
            result = runner.invoke(init, ['.'], input=inputs)

            assert result.exit_code == 0

            # Check stepA config
            with open('stepA/config.yaml', 'r') as f:
                step_config_content = f.read()
                step_config = yaml.safe_load(step_config_content)

            assert 'pipeline_step' in step_config
            assert step_config['pipeline_step']['id'] == 'stepA'
            assert step_config['pipeline_step']['user_id'] == 'test-user'
            assert step_config['pipeline_step']['app_id'] == 'test-app'
            assert 'pipeline_step_input_params' in step_config
            assert 'build_info' in step_config
            assert 'pipeline_step_compute_info' in step_config

            # Ensure no TODO comments exist
            assert 'TODO' not in step_config_content

    def test_init_command_includes_helpful_messages(self):
        """Test that init command works successfully."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            inputs = "test-user\ntest-app\ntest-pipeline\n2\nstepA\nstepB\n"
            result = runner.invoke(init, ['.'], input=inputs)

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

    def test_init_command_creates_workflow_arguments_template(self):
        """Test that the generated pipeline config includes workflow-level arguments template."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            inputs = "test-user\ntest-app\ntest-pipeline\n2\nstepA\nstepB\n"
            result = runner.invoke(init, ['.'], input=inputs)

            assert result.exit_code == 0

            # Load and validate the generated config
            with open('config.yaml', 'r') as f:
                config_content = f.read()
                config = yaml.safe_load(config_content)

            # Check that argo orchestration spec has workflow arguments
            argo_spec = config['pipeline']['orchestration_spec']['argo_orchestration_spec']
            parsed_argo = yaml.safe_load(argo_spec)

            # Verify workflow-level arguments exist
            assert 'spec' in parsed_argo
            assert 'arguments' in parsed_argo['spec']
            assert 'parameters' in parsed_argo['spec']['arguments']

            # Check that template reference is used in step parameters
            step_templates = parsed_argo['spec']['templates']
            sequence_template = next(t for t in step_templates if t['name'] == 'sequence')

            # Find a step that uses workflow parameters
            step_found = False
            for step_group in sequence_template['steps']:
                for step in step_group:
                    if 'arguments' in step and 'parameters' in step['arguments']:
                        for param in step['arguments']['parameters']:
                            if 'value' in param and '{{workflow.parameters.' in param['value']:
                                step_found = True
                                break

            assert step_found, (
                "Expected to find template references to workflow parameters in step arguments"
            )

    def test_init_command_with_template_option(self):
        """Test that init command works with --template option."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            # Mock the helper functions
            with (
                patch('clarifai.cli.pipeline._prepare_pipeline_path') as mock_prepare,
                patch('clarifai.cli.pipeline._init_from_template') as mock_template_init,
                patch('clarifai.cli.pipeline._show_completion_message') as mock_completion,
            ):
                mock_prepare.return_value = '/test/path'
                mock_template_init.return_value = True

                result = runner.invoke(init, ['--template', 'image-classification', '.'])

                # Should call template initialization with prepared path
                mock_prepare.assert_called_once_with('.', 'image-classification')
                mock_template_init.assert_called_once_with('/test/path', 'image-classification')
                mock_completion.assert_called_once_with('/test/path')

    def test_init_command_without_template_calls_interactive(self):
        """Test that init command without --template calls interactive flow."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            # Mock helper functions
            with (
                patch('clarifai.cli.pipeline._prepare_pipeline_path') as mock_prepare,
                patch('clarifai.cli.pipeline._init_interactive') as mock_interactive,
                patch('clarifai.cli.pipeline._init_from_template') as mock_template,
                patch('clarifai.cli.pipeline._show_completion_message') as mock_completion,
            ):
                mock_prepare.return_value = '/test/path'
                mock_interactive.return_value = True

                result = runner.invoke(init, ['.'])

                # Should call interactive initialization with prepared path
                mock_prepare.assert_called_once_with('.', None)
                mock_interactive.assert_called_once_with('/test/path')
                mock_template.assert_not_called()
                mock_completion.assert_called_once_with('/test/path')

    @patch('clarifai.utils.template_manager.TemplateManager')
    def test_init_from_template_success(self, mock_template_manager_class):
        """Test successful template-based initialization with generic template parameters."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        # Mock template info with generic parameter structure for testing
        mock_info = {
            'name': 'test-template',
            'type': 'train',
            'step_directories': ['StepA', 'StepB'],
            'parameters': [
                {
                    'name': 'EXAMPLE_PATH',
                    'default_value': '/default/path',
                },
                {
                    'name': 'EXAMPLE_SIZE',
                    'default_value': 16,
                },
            ],
            'config': {'pipeline': {'id': 'test-template'}},
        }
        mock_manager.get_template_info.return_value = mock_info
        mock_manager.copy_template.return_value = True

        with runner.isolated_filesystem():
            # Import the function to test directly
            from clarifai.cli.pipeline import _init_from_template

            # Mock click.prompt to simulate user input
            with patch('click.prompt') as mock_prompt:
                # Setup prompt responses
                mock_prompt.side_effect = [
                    'test-user',  # User ID
                    'test-app',  # App ID
                    'my-pipeline',  # Pipeline ID
                    '/custom/path',  # Example Path parameter
                    '32',  # Example Size parameter
                ]

                result = _init_from_template('/test/path', 'test-template')

                # Verify template manager was called correctly
                mock_manager.get_template_info.assert_called_once_with('test-template')
                mock_manager.copy_template.assert_called_once()

                # Check that substitutions were passed
                call_args = mock_manager.copy_template.call_args
                assert call_args[0][0] == 'test-template'  # template name
                assert call_args[0][1] == '/test/path'  # destination path

                substitutions = call_args[0][2]  # substitutions dict
                assert 'user_id' in substitutions  # Basic substitutions
                assert 'app_id' in substitutions
                assert 'id' in substitutions
                assert substitutions['user_id'] == 'test-user'
                assert substitutions['app_id'] == 'test-app'
                assert substitutions['id'] == 'my-pipeline'

                # Should return True for success
                assert result is True

    @patch('clarifai.utils.template_manager.TemplateManager')
    def test_init_from_template_not_found(self, mock_template_manager_class):
        """Test template initialization when template not found."""
        # Mock template manager to return None (template not found)
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager
        mock_manager.get_template_info.return_value = None

        # Import the function to test directly
        from clarifai.cli.pipeline import _init_from_template

        result = _init_from_template('/test/path', 'nonexistent-template')

        # Should return False for failure
        assert result is False
        mock_manager.get_template_info.assert_called_once_with('nonexistent-template')

    @patch('clarifai.utils.template_manager.TemplateManager')
    def test_init_from_template_with_copy_failure(self, mock_template_manager_class):
        """Test template initialization when copying fails."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        # Mock template info but copy failure
        mock_info = {
            'name': 'test-template',
            'type': 'train',
            'step_directories': ['StepA'],
            'parameters': [],
            'config': {'pipeline': {'id': 'test-template'}},
        }
        mock_manager.get_template_info.return_value = mock_info
        mock_manager.copy_template.return_value = False  # Copy fails

        # Import the function to test directly
        from clarifai.cli.pipeline import _init_from_template

        # Mock click.prompt for basic inputs
        with patch('click.prompt') as mock_prompt:
            mock_prompt.side_effect = ['test-user', 'test-app', 'my-pipeline']

            result = _init_from_template('/test/path', 'test-template')

            # Should return False for copy failure
            assert result is False

    @patch('clarifai.utils.template_manager.TemplateManager')
    def test_init_from_template_with_parameters(self, mock_template_manager_class):
        """Test template initialization with multiple generic parameters."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        # Mock template with multiple generic parameters for testing
        mock_info = {
            'name': 'complex-template',
            'type': 'train',
            'step_directories': ['LoadStep', 'ProcessStep', 'SaveStep'],
            'parameters': [
                {
                    'name': 'PARAM_A',
                    'default_value': '/default/input',
                },
                {
                    'name': 'PARAM_B',
                    'default_value': '/default/output',
                },
                {
                    'name': 'PARAM_C',
                    'default_value': 'default_value',
                },
                {
                    'name': 'PARAM_D',
                    'default_value': 0.001,
                },
            ],
            'config': {'pipeline': {'id': 'complex-template'}},
        }
        mock_manager.get_template_info.return_value = mock_info
        mock_manager.copy_template.return_value = True

        # Import the function to test directly
        from clarifai.cli.pipeline import _init_from_template

        # Mock click.prompt for all inputs
        with patch('click.prompt') as mock_prompt:
            mock_prompt.side_effect = [
                'user',  # User ID
                'app',  # App ID
                'pipeline',  # Pipeline ID
                '/new/input',  # Param A
                '/new/output',  # Param B
                'new_value',  # Param C
                '0.002',  # Param D
            ]

            result = _init_from_template('/test/path', 'complex-template')

            # Verify all parameters were processed
            call_args = mock_manager.copy_template.call_args
            substitutions = call_args[0][2]

            # Verify the function succeeded
            assert result is True

            # Verify copy_template was called
            assert mock_manager.copy_template.called
            call_args = mock_manager.copy_template.call_args
            substitutions = call_args[0][2]

            # Check that basic substitutions are present
            assert 'user_id' in substitutions
            assert 'app_id' in substitutions
            assert 'id' in substitutions
            assert substitutions['user_id'] == 'user'
            assert substitutions['app_id'] == 'app'
            assert substitutions['id'] == 'pipeline'

            # Check that parameter substitutions are present (only if different from default)
            # Since all inputs differ from defaults, they should be in substitutions
            assert 'PARAM_A' in substitutions
            assert substitutions['PARAM_A'] == '/new/input'
            assert 'PARAM_C' in substitutions
            assert substitutions['PARAM_C'] == 'new_value'
            assert 'PARAM_D' in substitutions
            assert substitutions['PARAM_D'] == '0.002'

    @patch('clarifai.utils.template_manager.TemplateManager')
    def test_init_from_template_custom_pipeline_id(self, mock_template_manager_class):
        """Test template initialization with custom pipeline ID substitution."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        mock_info = {
            'name': 'original-template',
            'type': 'train',
            'step_directories': ['Step1'],
            'parameters': [],
            'config': {'pipeline': {'id': 'original-template'}},
        }
        mock_manager.get_template_info.return_value = mock_info
        mock_manager.copy_template.return_value = True

        # Import the function to test directly
        from clarifai.cli.pipeline import _init_from_template

        # Mock click.prompt for inputs with custom pipeline name
        with patch('click.prompt') as mock_prompt:
            mock_prompt.side_effect = [
                'user',  # User ID
                'app',  # App ID
                'custom-pipeline-name',  # Custom Pipeline ID
            ]

            result = _init_from_template('/test/path', 'original-template')

            # Verify pipeline ID substitution was included
            call_args = mock_manager.copy_template.call_args
            substitutions = call_args[0][2]

            # The new system stores basic fields directly
            assert 'user_id' in substitutions
            assert 'app_id' in substitutions
            assert 'id' in substitutions
            assert substitutions['id'] == 'custom-pipeline-name'

            # Should return True for success
            assert result is True


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
            mock_pipeline.run.assert_called_once_with(
                timeout=300, monitor_interval=5, input_args_override=None
            )

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
            mock_pipeline.run.assert_called_once_with(
                timeout=3600, monitor_interval=10, input_args_override=None
            )

    def test_run_command_missing_required_args(self):
        """Test that run command fails when required arguments are missing."""
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
            mock_pipeline.run.assert_called_once_with(
                timeout=3600, monitor_interval=10, input_args_override=None
            )

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
    @patch('clarifai.client.user.User')
    @patch('clarifai.cli.pipeline.display_co_resources')
    def test_list_command_success_no_app_id(self, mock_display, mock_user_class, mock_validate):
        """Test that list command works without app_id (lists across all apps)."""
        # Setup mocks
        mock_validate.return_value = None
        mock_user_instance = Mock()
        mock_user_class.return_value = mock_user_instance
        mock_user_instance.list_pipelines.return_value = [
            {
                'id': 'pipeline1',
                'user_id': 'user1',
                'app_id': 'app1',
                'description': 'Test pipeline 1',
            },
            {
                'id': 'pipeline2',
                'user_id': 'user1',
                'app_id': 'app2',
                'description': 'Test pipeline 2',
            },
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
            user_id='test-user', pat='test-pat', base_url='https://api.clarifai.com'
        )
        mock_user_instance.list_pipelines.assert_called_once_with(page_no=1, per_page=10)
        mock_display.assert_called_once()

    @patch('clarifai.cli.pipeline.validate_context')
    @patch('clarifai.client.app.App')
    @patch('clarifai.cli.pipeline.display_co_resources')
    def test_list_command_success_with_app_id(self, mock_display, mock_app_class, mock_validate):
        """Test that list command works with app_id (lists within specific app)."""
        # Setup mocks
        mock_validate.return_value = None
        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance
        mock_app_instance.list_pipelines.return_value = [
            {
                'id': 'pipeline1',
                'user_id': 'user1',
                'app_id': 'app1',
                'description': 'Test pipeline 1',
            },
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
            base_url='https://api.clarifai.com',
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

        with patch('clarifai.client.user.User') as mock_user_class:
            mock_user_instance = Mock()
            mock_user_class.return_value = mock_user_instance
            mock_user_instance.list_pipelines.return_value = []

            with patch('clarifai.cli.pipeline.display_co_resources') as mock_display:
                result = runner.invoke(list_command, [], obj=ctx_obj)

                assert result.exit_code == 0
                mock_user_instance.list_pipelines.assert_called_once_with(page_no=1, per_page=16)


class TestPipelineTemplateCommands:
    """Test cases for the pipeline template CLI commands."""

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_list_templates_command_all_templates(self, mock_template_manager_class):
        """Test that list_templates command works without type filter."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        # Mock template list
        mock_templates = [
            {'name': 'template1', 'type': 'train', 'description': 'Training template'},
            {'name': 'template2', 'type': 'data', 'description': 'Data processing template'},
        ]
        mock_manager.list_templates.return_value = mock_templates

        runner = CliRunner()
        result = runner.invoke(list_templates, [])

        assert result.exit_code == 0
        mock_manager.list_templates.assert_called_once_with(None)
        # Should display both templates
        assert 'template1' in result.output
        assert 'template2' in result.output

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_list_templates_command_with_type_filter(self, mock_template_manager_class):
        """Test that list_templates command works with type filter."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        # Mock filtered template list
        mock_templates = [
            {'name': 'train-template1', 'type': 'train', 'description': 'Training template 1'},
            {'name': 'train-template2', 'type': 'train', 'description': 'Training template 2'},
        ]
        mock_manager.list_templates.return_value = mock_templates

        runner = CliRunner()
        result = runner.invoke(list_templates, ['--type', 'train'])

        assert result.exit_code == 0
        mock_manager.list_templates.assert_called_once_with('train')
        # Should display only train templates
        assert 'train-template1' in result.output
        assert 'train-template2' in result.output

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_list_templates_command_empty_list(self, mock_template_manager_class):
        """Test that list_templates command handles empty template list."""
        # Mock template manager to return empty list
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager
        mock_manager.list_templates.return_value = []

        runner = CliRunner()
        result = runner.invoke(list_templates, [])

        assert result.exit_code == 0
        mock_manager.list_templates.assert_called_once_with(None)
        # Should display message about no templates
        assert 'No templates found' in result.output or len(result.output.strip()) == 0

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_info_command_template_found(self, mock_template_manager_class):
        """Test that info command works when template exists."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        # Mock template info with new parameter structure
        mock_info = {
            'name': 'test-template',
            'type': 'train',
            'path': '/path/to/test-template',
            'description': 'A test training template',
            'step_directories': ['LoadData', 'TrainModel', 'SaveModel'],
            'parameters': [
                {
                    'name': 'EXAMPLE_PATH',
                    'default_value': '/default/data/path',
                },
                {
                    'name': 'EXAMPLE_TYPE',
                    'default_value': 'default_type',
                },
            ],
            'config': {'pipeline': {'id': 'test-template'}},
        }
        mock_manager.get_template_info.return_value = mock_info

        runner = CliRunner()
        result = runner.invoke(info, ['test-template'])

        assert result.exit_code == 0
        mock_manager.get_template_info.assert_called_once_with('test-template')

        # Should display template information
        assert 'test-template' in result.output
        assert 'train' in result.output
        assert 'LoadData' in result.output
        assert 'TrainModel' in result.output
        assert 'SaveModel' in result.output
        assert 'EXAMPLE_PATH (default: /default/data/path)' in result.output
        assert 'EXAMPLE_TYPE (default: default_type)' in result.output

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_info_command_template_not_found(self, mock_template_manager_class):
        """Test that info command handles template not found."""
        # Mock template manager to return None
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager
        mock_manager.get_template_info.return_value = None

        runner = CliRunner()
        result = runner.invoke(info, ['nonexistent-template'])

        assert result.exit_code == 0
        mock_manager.get_template_info.assert_called_once_with('nonexistent-template')

        # Should display error message
        assert 'not found' in result.output or 'Error' in result.output

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_info_command_template_with_no_parameters(self, mock_template_manager_class):
        """Test that info command handles templates with no parameters."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        # Mock template info without parameters
        mock_info = {
            'name': 'simple-template',
            'type': 'data',
            'path': '/path/to/simple-template',
            'description': 'A simple data processing template',
            'step_directories': ['ProcessData'],
            'parameters': [],
            'config': {'pipeline': {'id': 'simple-template'}},
        }
        mock_manager.get_template_info.return_value = mock_info

        runner = CliRunner()
        result = runner.invoke(info, ['simple-template'])

        assert result.exit_code == 0
        mock_manager.get_template_info.assert_called_once_with('simple-template')

        # Should display template information but mention no parameters
        assert 'simple-template' in result.output
        assert 'data' in result.output
        assert 'ProcessData' in result.output
        assert (
            'No template parameters found' in result.output
            or 'Parameters' not in result.output
            or result.output.count('Parameters') <= 1
        )  # Just the section header

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_list_templates_command_with_rich_display(self, mock_template_manager_class):
        """Test that list_templates displays templates in a formatted table."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        # Mock template list with varied data
        mock_templates = [
            {
                'name': 'image-classification',
                'type': 'train',
                'description': 'Image classification training',
            },
            {'name': 'text-prep', 'type': 'data', 'description': 'Text preprocessing'},
            {'name': 'model-deploy', 'type': 'deploy', 'description': 'Model deployment'},
        ]
        mock_manager.list_templates.return_value = mock_templates

        runner = CliRunner()
        result = runner.invoke(list_templates, [])

        assert result.exit_code == 0

        # Should display all template information
        assert 'image-classification' in result.output
        assert 'text-prep' in result.output
        assert 'model-deploy' in result.output
        assert 'train' in result.output
        assert 'data' in result.output
        assert 'deploy' in result.output

    def test_info_command_help_text(self):
        """Test that info command shows proper help text."""
        runner = CliRunner()
        result = runner.invoke(info, ['--help'])

        assert result.exit_code == 0
        assert 'Show detailed information about a specific template' in result.output
        assert 'TEMPLATE_NAME' in result.output

    def test_list_templates_command_help_text(self):
        """Test that list_templates command shows proper help text."""
        runner = CliRunner()
        result = runner.invoke(list_templates, ['--help'])

        assert result.exit_code == 0
        assert (
            'List available pipeline templates' in result.output
            or 'Lists all available pipeline templates' in result.output
        )
        assert '--type' in result.output
