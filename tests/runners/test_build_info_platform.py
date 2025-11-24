"""Tests for build_info.platform configuration in model upload."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from clarifai.runners.models.model_builder import ModelBuilder


@pytest.fixture
def model_with_platform_path(tmp_path):
    """Create a test model with platform specified in build_info."""
    model_path = tmp_path / "platform_model"
    model_path.mkdir()

    # Create version folder
    version_path = model_path / "1"
    version_path.mkdir()

    # Create config.yaml with platform in build_info
    config_content = """
model:
  id: "test-platform-model"
  model_type_id: "text-to-text"
  user_id: "test_user"
  app_id: "test_app"

build_info:
  python_version: "3.12"
  platform: "linux/amd64,linux/arm64"

inference_compute_info:
  cpu_limit: "1"
  cpu_memory: "1Gi"
  num_accelerators: 0
"""
    with open(model_path / "config.yaml", "w") as f:
        f.write(config_content)

    # Create requirements.txt
    with open(model_path / "requirements.txt", "w") as f:
        f.write("clarifai>=11.9.0\n")

    # Create model.py
    model_content = """
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Text

class MyModel(ModelClass):
    def load_model(self):
        pass

    @ModelClass.method
    def predict(self, text1: Text = "") -> Text:
        return Text(text1.text + "Hello")

    def test(self):
        res = self.predict(Text("test"))
        assert res.text == "testHello"
"""
    with open(version_path / "model.py", "w") as f:
        f.write(model_content)

    return str(model_path)


@pytest.fixture
def model_without_platform_path(tmp_path):
    """Create a test model without platform specified in build_info."""
    model_path = tmp_path / "no_platform_model"
    model_path.mkdir()

    # Create version folder
    version_path = model_path / "1"
    version_path.mkdir()

    # Create config.yaml without platform
    config_content = """
model:
  id: "test-no-platform-model"
  model_type_id: "text-to-text"
  user_id: "test_user"
  app_id: "test_app"

build_info:
  python_version: "3.12"

inference_compute_info:
  cpu_limit: "1"
  cpu_memory: "1Gi"
  num_accelerators: 0
"""
    with open(model_path / "config.yaml", "w") as f:
        f.write(config_content)

    # Create requirements.txt
    with open(model_path / "requirements.txt", "w") as f:
        f.write("clarifai>=11.9.0\n")

    # Create model.py
    model_content = """
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Text

class MyModel(ModelClass):
    def load_model(self):
        pass

    @ModelClass.method
    def predict(self, text1: Text = "") -> Text:
        return Text(text1.text + "Hello")

    def test(self):
        res = self.predict(Text("test"))
        assert res.text == "testHello"
"""
    with open(version_path / "model.py", "w") as f:
        f.write(model_content)

    return str(model_path)


class TestBuildInfoPlatform:
    """Test build_info.platform configuration."""

    def test_platform_config_read(self, model_with_platform_path):
        """Test that platform is correctly read from config.yaml."""
        builder = ModelBuilder(model_with_platform_path, download_validation_only=True)

        # Check that platform is in the config
        assert 'build_info' in builder.config
        assert 'platform' in builder.config['build_info']
        assert builder.config['build_info']['platform'] == "linux/amd64,linux/arm64"

    def test_no_platform_config(self, model_without_platform_path):
        """Test that models without platform configuration still work."""
        builder = ModelBuilder(model_without_platform_path, download_validation_only=True)

        # Check that build_info exists but platform doesn't
        assert 'build_info' in builder.config
        assert 'platform' not in builder.config['build_info']

    @patch('clarifai.runners.models.model_builder.logger')
    def test_platform_set_on_build_info_when_available(
        self, mock_logger, model_with_platform_path
    ):
        """Test that platform is set on BuildInfo when the field is available."""
        from clarifai_grpc.grpc.api import resources_pb2

        builder = ModelBuilder(model_with_platform_path, download_validation_only=True)

        # Mock get_method_signatures to avoid loading the model
        with patch.object(builder, 'get_method_signatures', return_value=[]):
            # Create a real BuildInfo and check if it has platform field
            build_info = resources_pb2.BuildInfo()

            if hasattr(build_info, 'platform'):
                # If platform field exists, test that it gets set
                model_version_proto = builder.get_model_version_proto()
                assert model_version_proto.build_info.platform == "linux/amd64,linux/arm64"
                # Verify that the info message was logged
                mock_logger.info.assert_called()
                info_calls = [str(call) for call in mock_logger.info.call_args_list]
                assert any('platform' in str(call).lower() for call in info_calls)
            else:
                # If platform field doesn't exist yet, just verify warning is logged
                model_version_proto = builder.get_model_version_proto()
                # Verify that a warning was logged since field doesn't exist
                mock_logger.warning.assert_called_once()
                warning_message = mock_logger.warning.call_args[0][0]
                assert 'platform' in warning_message.lower()

    @patch('clarifai.runners.models.model_builder.logger')
    def test_platform_warning_when_field_not_available(
        self, mock_logger, model_with_platform_path
    ):
        """Test that a warning is logged when platform field is not available in BuildInfo."""
        from clarifai_grpc.grpc.api import resources_pb2

        builder = ModelBuilder(model_with_platform_path, download_validation_only=True)

        # Mock get_method_signatures to avoid loading the model
        with patch.object(builder, 'get_method_signatures', return_value=[]):
            # Create a real BuildInfo and check if it has platform field
            build_info = resources_pb2.BuildInfo()

            if not hasattr(build_info, 'platform'):
                # Platform field doesn't exist yet - verify warning is logged
                model_version_proto = builder.get_model_version_proto()

                # Verify that a warning was logged
                mock_logger.warning.assert_called_once()
                warning_message = mock_logger.warning.call_args[0][0]
                assert 'platform' in warning_message.lower()
                assert 'not supported' in warning_message.lower()
            else:
                # If platform field exists, skip this test as it's not relevant
                pytest.skip("Platform field is available in current clarifai-grpc version")

    def test_platform_yaml_format(self, model_with_platform_path):
        """Test that platform value is correctly formatted in YAML."""
        config_file = Path(model_with_platform_path) / "config.yaml"
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Verify platform is a string (not a list or other type)
        platform = config['build_info']['platform']
        assert isinstance(platform, str)
        # Verify it contains both platforms
        assert 'linux/amd64' in platform
        assert 'linux/arm64' in platform

    def test_single_platform_format(self, tmp_path):
        """Test that single platform values are also supported."""
        model_path = tmp_path / "single_platform_model"
        model_path.mkdir()
        version_path = model_path / "1"
        version_path.mkdir()

        # Create config with single platform
        config_content = """
model:
  id: "test-single-platform"
  model_type_id: "text-to-text"
  user_id: "test_user"
  app_id: "test_app"

build_info:
  python_version: "3.12"
  platform: "linux/amd64"

inference_compute_info:
  cpu_limit: "1"
  cpu_memory: "1Gi"
  num_accelerators: 0
"""
        with open(model_path / "config.yaml", "w") as f:
            f.write(config_content)

        with open(model_path / "requirements.txt", "w") as f:
            f.write("clarifai>=11.9.0\n")

        model_content = """
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Text

class MyModel(ModelClass):
    def load_model(self):
        pass

    @ModelClass.method
    def predict(self, text1: Text = "") -> Text:
        return Text("test")
"""
        with open(version_path / "model.py", "w") as f:
            f.write(model_content)

        builder = ModelBuilder(str(model_path), download_validation_only=True)
        assert builder.config['build_info']['platform'] == "linux/amd64"
