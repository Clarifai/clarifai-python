import os
import tempfile

import pytest
import yaml

from clarifai.runners.pipelines.pipeline_builder import PipelineBuilder


class TestPipelineLockfile:
    """Test cases for pipeline lockfile functionality."""

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

    def test_generate_lockfile_data(self, temp_config_file):
        """Test generating lockfile data."""
        builder = PipelineBuilder(temp_config_file)
        builder.uploaded_step_versions = {"stepA": "version-123"}

        lockfile_data = builder.generate_lockfile_data(
            pipeline_id="test-pipeline", pipeline_version_id="pipeline-version-456"
        )

        # Verify lockfile structure
        assert "pipeline" in lockfile_data
        pipeline_data = lockfile_data["pipeline"]

        assert pipeline_data["id"] == "test-pipeline"
        assert pipeline_data["user_id"] == "test-user"
        assert pipeline_data["app_id"] == "test-app"
        assert pipeline_data["version_id"] == "pipeline-version-456"
        assert "orchestration_spec" in pipeline_data

        # Verify that templateRef was updated with version
        argo_spec_str = pipeline_data["orchestration_spec"]["argo_orchestration_spec"]
        argo_spec = yaml.safe_load(argo_spec_str)
        template_ref = argo_spec["spec"]["templates"][0]["steps"][0][0]["templateRef"]
        expected_name = "users/test-user/apps/test-app/pipeline_steps/stepA/versions/version-123"
        assert template_ref["name"] == expected_name
        assert template_ref["template"] == expected_name

    def test_save_lockfile(self, temp_config_file):
        """Test saving lockfile to disk."""
        builder = PipelineBuilder(temp_config_file)
        builder.uploaded_step_versions = {"stepA": "version-123"}

        lockfile_data = builder.generate_lockfile_data(
            pipeline_id="test-pipeline", pipeline_version_id="pipeline-version-456"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            lockfile_path = os.path.join(temp_dir, "config-lock.yaml")
            builder.save_lockfile(lockfile_data, lockfile_path)

            # Verify file was created and contains correct data
            assert os.path.exists(lockfile_path)

            with open(lockfile_path, 'r') as f:
                saved_data = yaml.safe_load(f)

            assert saved_data["pipeline"]["id"] == "test-pipeline"
            assert saved_data["pipeline"]["version_id"] == "pipeline-version-456"

    def test_generate_lockfile_data_no_step_versions(self, temp_config_file):
        """Test generating lockfile data when no step versions are uploaded."""
        builder = PipelineBuilder(temp_config_file)
        # Don't set uploaded_step_versions

        lockfile_data = builder.generate_lockfile_data(
            pipeline_id="test-pipeline", pipeline_version_id="pipeline-version-456"
        )

        # Should still generate lockfile data with basic info
        assert "pipeline" in lockfile_data
        pipeline_data = lockfile_data["pipeline"]

        assert pipeline_data["id"] == "test-pipeline"
        assert pipeline_data["user_id"] == "test-user"
        assert pipeline_data["app_id"] == "test-app"
        assert pipeline_data["version_id"] == "pipeline-version-456"
