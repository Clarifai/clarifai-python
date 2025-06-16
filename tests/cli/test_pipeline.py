import os
import tempfile
import pytest
import yaml
from unittest.mock import Mock, patch
from click.testing import CliRunner

from clarifai.runners.pipelines.pipeline_builder import (
    PipelineBuilder,
    PipelineConfigValidator,
    upload_pipeline
)
from clarifai.cli.pipeline import upload


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
        config = {
            "pipeline": {
                "id": "",
                "user_id": "test-user",
                "app_id": "test-app"
            }
        }
        
        with pytest.raises(ValueError, match="'id' cannot be empty"):
            PipelineConfigValidator.validate_config(config)
    
    def test_validate_config_missing_orchestration_spec(self):
        """Test validation with missing orchestration spec."""
        config = {
            "pipeline": {
                "id": "test-pipeline",
                "user_id": "test-user",
                "app_id": "test-app"
            }
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
                "orchestration_spec": {
                    "argo_orchestration_spec": "invalid: yaml: :"
                }
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
                }
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
                }
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
                }
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
                }
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
                }
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
    
    @patch('clarifai.runners.pipelines.pipeline_builder.PipelineBuilder._upload_pipeline_step_with_version_capture')
    def test_upload_pipeline_steps_success(self, mock_upload, temp_config_file):
        """Test successful pipeline steps upload."""
        mock_upload.return_value = (True, "version-123")
        
        builder = PipelineBuilder(temp_config_file)
        
        # Mock the directory existence
        with patch('os.path.exists', return_value=True):
            result = builder.upload_pipeline_steps()
        
        assert result is True
        assert builder.uploaded_step_versions == {"stepA": "version-123"}
    
    @patch('clarifai.runners.pipelines.pipeline_builder.PipelineBuilder._upload_pipeline_step_with_version_capture')
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
        assert "CONFIG_PATH" in result.output
    
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
                }
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
                }
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
                }
            }
        }
        
        with pytest.raises(ValueError, match="argo_orchestration_spec must have apiVersion 'argoproj.io/v1alpha1'"):
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
                }
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
                }
            }
        }
        
        with pytest.raises(ValueError, match="templateRef must have both 'name' and 'template' fields"):
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
                }
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
                }
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