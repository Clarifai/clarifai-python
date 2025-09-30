import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml

from clarifai.runners.pipeline_steps.pipeline_step_builder import (
    PipelineStepBuilder,
    upload_pipeline_step,
)


# Mock BaseClient at module level to avoid auth issues during initialization
@patch('clarifai.runners.pipeline_steps.pipeline_step_builder.BaseClient')
class TestPipelineStepBuilder:
    """Test cases for PipelineStepBuilder."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def valid_config(self):
        """Return a valid configuration for testing."""
        return {
            "pipeline_step": {"id": "test-step", "user_id": "test-user", "app_id": "test-app"},
            "pipeline_step_compute_info": {
                "cpu_limit": "1000m",
                "cpu_memory": "1Gi",
                "num_accelerators": 1,
            },
            "pipeline_step_input_params": [
                {
                    "name": "test_param",
                    "default": "default_value",
                    "description": "Test parameter",
                    "accepted_values": ["value1", "value2"],
                }
            ],
            "build_info": {"python_version": "3.12"},
        }

    @pytest.fixture
    def setup_test_folder(self, temp_dir, valid_config):
        """Set up a test folder with required files."""
        # Create config.yaml
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)

        # Create 1/ subdirectory and pipeline_step.py
        os.makedirs(os.path.join(temp_dir, "1"))
        pipeline_step_path = os.path.join(temp_dir, "1", "pipeline_step.py")
        with open(pipeline_step_path, 'w') as f:
            f.write("# Pipeline step implementation")

        # Create requirements.txt
        requirements_path = os.path.join(temp_dir, "requirements.txt")
        with open(requirements_path, 'w') as f:
            f.write("requests==2.32.0\n")

        return temp_dir

    def test_init_valid_folder(self, mock_base_client, setup_test_folder, valid_config):
        """Test successful initialization with valid folder."""
        builder = PipelineStepBuilder(setup_test_folder)

        assert builder.folder == os.path.abspath(setup_test_folder)
        assert builder.config == valid_config
        assert builder.pipeline_step_id == "test-step"
        assert builder.pipeline_step_proto.id == "test-step"
        assert builder.pipeline_step_proto.user_id == "test-user"

    def test_init_missing_config(self, mock_base_client, temp_dir):
        """Test initialization fails with missing config.yaml."""
        with pytest.raises(FileNotFoundError, match="Required file 'config.yaml' not found"):
            PipelineStepBuilder(temp_dir)

    def test_init_missing_pipeline_step_file(self, mock_base_client, temp_dir, valid_config):
        """Test initialization fails with missing pipeline_step.py."""
        # Create config.yaml but not the pipeline_step.py
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)

        with pytest.raises(
            FileNotFoundError, match="Required file '1/pipeline_step.py' not found"
        ):
            PipelineStepBuilder(temp_dir)

    def test_validate_config_missing_pipeline_step_section(self, mock_base_client, temp_dir):
        """Test validation fails with missing pipeline_step section."""
        config = {"some_other_section": {}}
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Create the pipeline_step.py file
        os.makedirs(os.path.join(temp_dir, "1"))
        pipeline_step_path = os.path.join(temp_dir, "1", "pipeline_step.py")
        with open(pipeline_step_path, 'w') as f:
            f.write("# Pipeline step implementation")

        with pytest.raises(ValueError, match="pipeline_step section not found in config.yaml"):
            PipelineStepBuilder(temp_dir)

    def test_validate_config_missing_required_fields(self, mock_base_client, temp_dir):
        """Test validation fails with missing required fields."""
        config = {
            "pipeline_step": {
                "id": "test-step"
                # Missing user_id and app_id
            },
            "pipeline_step_compute_info": {},
        }
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Create the pipeline_step.py file
        os.makedirs(os.path.join(temp_dir, "1"))
        pipeline_step_path = os.path.join(temp_dir, "1", "pipeline_step.py")
        with open(pipeline_step_path, 'w') as f:
            f.write("# Pipeline step implementation")

        with pytest.raises(ValueError, match="user_id not found in pipeline_step section"):
            PipelineStepBuilder(temp_dir)

    def test_validate_config_empty_required_fields(self, mock_base_client, temp_dir):
        """Test validation fails with empty required fields."""
        config = {
            "pipeline_step": {
                "id": "",  # Empty id
                "user_id": "test-user",
                "app_id": "test-app",
            },
            "pipeline_step_compute_info": {},
        }
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Create the pipeline_step.py file
        os.makedirs(os.path.join(temp_dir, "1"))
        pipeline_step_path = os.path.join(temp_dir, "1", "pipeline_step.py")
        with open(pipeline_step_path, 'w') as f:
            f.write("# Pipeline step implementation")

        with pytest.raises(ValueError, match="id cannot be empty in config.yaml"):
            PipelineStepBuilder(temp_dir)

    def test_validate_config_missing_compute_info(self, mock_base_client, temp_dir):
        """Test validation fails with missing pipeline_step_compute_info section."""
        config = {
            "pipeline_step": {"id": "test-step", "user_id": "test-user", "app_id": "test-app"}
            # Missing pipeline_step_compute_info
        }
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Create the pipeline_step.py file
        os.makedirs(os.path.join(temp_dir, "1"))
        pipeline_step_path = os.path.join(temp_dir, "1", "pipeline_step.py")
        with open(pipeline_step_path, 'w') as f:
            f.write("# Pipeline step implementation")

        with pytest.raises(ValueError, match="pipeline_step_compute_info section not found"):
            PipelineStepBuilder(temp_dir)

    def test_client_property(self, mock_base_client, setup_test_folder):
        """Test client property creates BaseClient correctly."""
        builder = PipelineStepBuilder(setup_test_folder)

        # Access client property
        client = builder.client

        # Verify BaseClient was called with correct parameters
        mock_base_client.assert_called_once_with(user_id="test-user", app_id="test-app")
        assert builder._client is not None

    def test_ensure_clarifai_requirement_adds_clarifai(self, mock_base_client, setup_test_folder):
        """Test that clarifai is added to requirements.txt if not present."""
        builder = PipelineStepBuilder(setup_test_folder)

        # Call the method
        builder._ensure_clarifai_requirement()

        # Check that clarifai was added
        requirements_path = os.path.join(setup_test_folder, "requirements.txt")
        with open(requirements_path, 'r') as f:
            content = f.read()

        assert "clarifai==" in content

    def test_ensure_clarifai_requirement_already_present(
        self, mock_base_client, setup_test_folder
    ):
        """Test that clarifai is not duplicated if already present."""
        builder = PipelineStepBuilder(setup_test_folder)

        # Add clarifai to requirements.txt first
        requirements_path = os.path.join(setup_test_folder, "requirements.txt")
        with open(requirements_path, 'w') as f:
            f.write("clarifai==10.0.0\nrequests==2.32.0\n")

        # Call the method
        builder._ensure_clarifai_requirement()

        # Check that clarifai wasn't duplicated
        with open(requirements_path, 'r') as f:
            content = f.read()

        clarifai_count = content.count("clarifai")
        assert clarifai_count == 1

    def test_ensure_clarifai_requirement_with_clarifai_grpc(
        self, mock_base_client, setup_test_folder
    ):
        """Test that clarifai is added when clarifai-grpc is present but clarifai is not.
        
        This test covers the bug where 'clarifai' in 'clarifai-grpc' was incorrectly
        detected as the clarifai package itself.
        """
        builder = PipelineStepBuilder(setup_test_folder)

        # Add clarifai-grpc and other packages to requirements.txt (realistic scenario)
        requirements_path = os.path.join(setup_test_folder, "requirements.txt")
        with open(requirements_path, 'w') as f:
            f.write("clarifai-grpc>=11.8.2\nclarifai-protocol>=0.0.32\nnumpy>=1.22.0\nrequests>=2.32.0\n")

        # Call the method
        builder._ensure_clarifai_requirement()

        # Check that clarifai was added
        with open(requirements_path, 'r') as f:
            content = f.read()

        # Verify clarifai was added (should contain both clarifai-grpc and clarifai)
        assert "clarifai-grpc" in content  # Should still have clarifai-grpc
        assert "clarifai==" in content  # Should have added clarifai
        
        # Make sure we have exactly one clarifai package (not matching clarifai-grpc)
        lines = content.strip().split('\n')
        clarifai_exact_lines = [line for line in lines if line.strip().startswith('clarifai==')]
        assert len(clarifai_exact_lines) == 1, f"Expected exactly 1 clarifai== line, got {len(clarifai_exact_lines)}"

    def test_create_dockerfile(self, mock_base_client, setup_test_folder):
        """Test Dockerfile creation."""
        builder = PipelineStepBuilder(setup_test_folder)

        # Call create_dockerfile
        builder.create_dockerfile()

        # Check that Dockerfile was created
        dockerfile_path = os.path.join(setup_test_folder, "Dockerfile")
        assert os.path.exists(dockerfile_path)

        # Check content
        with open(dockerfile_path, 'r') as f:
            content = f.read()

        assert "FROM --platform=$TARGETPLATFORM" in content
        assert "COPY --link requirements.txt" in content
        assert "COPY --link=true 1 /home/nonroot/main/1" in content
        assert "3.12" in content  # Python version from config

    def test_tar_file_property(self, mock_base_client, setup_test_folder):
        """Test tar_file property returns correct path."""
        builder = PipelineStepBuilder(setup_test_folder)

        expected_tar_file = f"{setup_test_folder}.tar.gz"
        assert builder.tar_file == expected_tar_file

    def test_get_pipeline_step_compute_info(self, mock_base_client, setup_test_folder):
        """Test compute info configuration parsing."""
        builder = PipelineStepBuilder(setup_test_folder)

        compute_info = builder.pipeline_step_compute_info

        assert compute_info.cpu_limit == "1000m"
        assert compute_info.cpu_memory == "1Gi"
        assert compute_info.num_accelerators == 1

    def test_load_config_invalid_yaml(self, mock_base_client, temp_dir):
        """Test that invalid YAML raises appropriate error."""
        # Create invalid YAML config
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML syntax

        # Create the pipeline_step.py file
        os.makedirs(os.path.join(temp_dir, "1"))
        pipeline_step_path = os.path.join(temp_dir, "1", "pipeline_step.py")
        with open(pipeline_step_path, 'w') as f:
            f.write("# Pipeline step implementation")

        with pytest.raises(ValueError, match="Error loading config.yaml"):
            PipelineStepBuilder(temp_dir)


# Mock PipelineStepBuilder to test the upload function
@patch('clarifai.runners.pipeline_steps.pipeline_step_builder.PipelineStepBuilder')
class TestUploadPipelineStep:
    """Test cases for the upload_pipeline_step function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def valid_config(self):
        """Return a valid configuration for testing."""
        return {
            "pipeline_step": {"id": "test-step", "user_id": "test-user", "app_id": "test-app"},
            "pipeline_step_compute_info": {},
        }

    @pytest.fixture
    def setup_test_folder(self, temp_dir, valid_config):
        """Set up a test folder with required files."""
        # Create config.yaml
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)

        # Create 1/ subdirectory and pipeline_step.py
        os.makedirs(os.path.join(temp_dir, "1"))
        pipeline_step_path = os.path.join(temp_dir, "1", "pipeline_step.py")
        with open(pipeline_step_path, 'w') as f:
            f.write("# Pipeline step implementation")

        return temp_dir

    @patch('builtins.input', return_value='')  # Mock user input
    def test_upload_pipeline_step_success(self, mock_input, mock_builder_class, setup_test_folder):
        """Test successful pipeline step upload."""
        # Mock builder instance
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.pipeline_step_id = "test-step"
        mock_builder.check_pipeline_step_exists.return_value = False
        mock_builder.upload_pipeline_step_version.return_value = True

        # Call upload function
        upload_pipeline_step(setup_test_folder, skip_dockerfile=True)

        # Verify builder was created and methods called
        mock_builder_class.assert_called_once_with(setup_test_folder)
        mock_builder.upload_pipeline_step_version.assert_called_once()
        mock_builder.create_dockerfile.assert_not_called()  # skip_dockerfile=True

    @patch('builtins.input', return_value='')  # Mock user input
    def test_upload_pipeline_step_with_dockerfile(
        self, mock_input, mock_builder_class, setup_test_folder
    ):
        """Test pipeline step upload with dockerfile creation."""
        # Mock builder instance
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.pipeline_step_id = "test-step"
        mock_builder.check_pipeline_step_exists.return_value = False
        mock_builder.upload_pipeline_step_version.return_value = True

        # Call upload function without skipping dockerfile
        upload_pipeline_step(setup_test_folder, skip_dockerfile=False)

        # Verify dockerfile creation was called
        mock_builder.create_dockerfile.assert_called_once()

    @patch('builtins.input', return_value='')  # Mock user input
    @patch('sys.exit')
    def test_upload_pipeline_step_failure(
        self, mock_exit, mock_input, mock_builder_class, setup_test_folder
    ):
        """Test pipeline step upload failure."""
        # Mock builder instance
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.pipeline_step_id = "test-step"
        mock_builder.check_pipeline_step_exists.return_value = False
        mock_builder.upload_pipeline_step_version.return_value = False

        # Call upload function
        upload_pipeline_step(setup_test_folder, skip_dockerfile=True)

        # Verify sys.exit was called with error code
        mock_exit.assert_called_once_with(1)

    @patch('builtins.input', return_value='')  # Mock user input
    def test_upload_pipeline_step_existing(
        self, mock_input, mock_builder_class, setup_test_folder
    ):
        """Test pipeline step upload when step already exists."""
        # Mock builder instance
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.pipeline_step_id = "test-step"
        mock_builder.check_pipeline_step_exists.return_value = True  # Already exists
        mock_builder.upload_pipeline_step_version.return_value = True

        # Call upload function
        upload_pipeline_step(setup_test_folder, skip_dockerfile=True)

        # Verify upload was still called (creates new version)
        mock_builder.upload_pipeline_step_version.assert_called_once()
