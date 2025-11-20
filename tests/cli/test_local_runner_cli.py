"""Tests for the local-runner CLI command.

These tests verify the basic functionality of the `clarifai model local-runner` command
by mocking external dependencies and testing key behaviors.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from clarifai.cli.base import cli


class TestLocalRunnerCLI:
    """Test cases for the local-runner CLI command."""

    @pytest.fixture
    def dummy_model_dir(self):
        """Create a dummy model directory structure for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create the model directory structure
            model_version_dir = os.path.join(tmp_dir, "1")
            os.makedirs(model_version_dir, exist_ok=True)

            # Create model.py
            model_py_content = """
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Text

class MyModel(ModelClass):
    def load_model(self):
        pass

    @ModelClass.method
    def predict(self, text1: Text = "") -> Text:
        return Text(text1.text + " Hello World")

    def test(self):
        res = self.predict(Text("test"))
        assert res.text == "test Hello World"
"""
            with open(os.path.join(model_version_dir, "model.py"), "w") as f:
                f.write(model_py_content)

            # Create config.yaml
            config = {
                "model": {
                    "id": "test-local-runner-model",
                    "user_id": "test-user",
                    "app_id": "test-app",
                    "model_type_id": "text-to-text",
                },
                "build_info": {"python_version": "3.11"},
                "inference_compute_info": {
                    "cpu_limit": "1",
                    "cpu_memory": "1Gi",
                    "num_accelerators": 0,
                },
            }
            with open(os.path.join(tmp_dir, "config.yaml"), "w") as f:
                yaml.dump(config, f)

            # Create requirements.txt
            with open(os.path.join(tmp_dir, "requirements.txt"), "w") as f:
                f.write("clarifai\n")

            yield tmp_dir

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    @patch("builtins.input")
    def test_local_runner_requires_installed_dependencies(
        self,
        mock_input,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """Test that local-runner checks for installed requirements."""
        # Setup: Requirements not installed
        mock_check_requirements.return_value = False
        mock_parse_requirements.return_value = []

        # Mock ModelBuilder to return a basic config
        mock_builder = MagicMock()
        mock_builder.config = {"model": {"model_type_id": "text-to-text"}, "toolkit": {}}
        mock_builder_class.return_value = mock_builder

        runner = CliRunner()
        runner.invoke(cli, ["login", "--user_id", "test-user", "--pat", "test-pat"])

        result = runner.invoke(
            cli,
            ["model", "local-runner", str(dummy_model_dir)],
        )

        # Should abort because requirements are not installed
        assert result.exit_code == 1
        mock_check_requirements.assert_called()

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.check_ollama_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_local_runner_checks_ollama_installation(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_ollama,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """Test that local-runner checks for Ollama when it's in dependencies."""
        # Setup: Requirements installed, but Ollama not installed
        mock_check_requirements.return_value = True
        mock_parse_requirements.return_value = ["ollama"]
        mock_check_ollama.return_value = False

        # Mock ModelBuilder
        mock_builder = MagicMock()
        mock_builder.config = {"model": {"model_type_id": "text-to-text"}, "toolkit": {}}
        mock_builder_class.return_value = mock_builder

        runner = CliRunner()
        runner.invoke(cli, ["login", "--user_id", "test-user", "--pat", "test-pat"])

        result = runner.invoke(
            cli,
            ["model", "local-runner", str(dummy_model_dir)],
        )

        # Should abort because Ollama is not installed
        assert result.exit_code == 1
        mock_check_ollama.assert_called()

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    @patch("builtins.input")
    def test_local_runner_user_declines_resource_creation(
        self,
        mock_input,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """Test that local-runner aborts when user declines resource creation."""
        # Setup: user declines resource creation
        mock_input.return_value = "n"  # User says no
        mock_check_requirements.return_value = True
        mock_parse_requirements.return_value = []

        # Mock ModelBuilder
        mock_builder = MagicMock()
        mock_builder.config = {"model": {"model_type_id": "text-to-text"}, "toolkit": {}}
        mock_builder_class.return_value = mock_builder
        mock_builder_class._load_config.return_value = mock_builder.config

        # Mock User that throws exception for missing compute cluster
        mock_user = MagicMock()
        mock_user.compute_cluster.side_effect = Exception("Cluster not found")
        mock_user_class.return_value = mock_user

        runner = CliRunner()
        runner.invoke(cli, ["login", "--user_id", "test-user", "--pat", "test-pat"])

        result = runner.invoke(
            cli,
            ["model", "local-runner", str(dummy_model_dir)],
        )

        # Should abort when user declines
        assert result.exit_code == 1
        # Verify that create_compute_cluster was NOT called
        mock_user.create_compute_cluster.assert_not_called()

    def test_local_runner_has_config_yaml_in_model_dir(self, dummy_model_dir):
        """Test that the dummy model directory contains a config.yaml file."""
        # This is a basic sanity test for the test fixture
        config_path = os.path.join(dummy_model_dir, "config.yaml")
        assert os.path.exists(config_path), "config.yaml should exist in the model directory"

        # Verify it can be loaded
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert 'model' in config
        assert config['model']['model_type_id'] == "text-to-text"
