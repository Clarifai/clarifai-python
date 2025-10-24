"""Tests for environment validation in model runners."""

from unittest.mock import patch

import pytest

from clarifai.runners.models.model_run_locally import ModelRunLocally, main


@pytest.fixture
def gpu_model_path(tmp_path):
    """Create a test model that requires GPU support."""
    model_path = tmp_path / "gpu_model"
    model_path.mkdir()

    # Create version folder
    version_path = model_path / "1"
    version_path.mkdir()

    # Create config.yaml requiring GPU
    config_content = """
model:
  id: "test-gpu-model"
  model_type_id: "text-to-text"
  user_id: "test_user"
  app_id: "test_app"

build_info:
  python_version: "3.12"

inference_compute_info:
  cpu_limit: "1"
  cpu_memory: "1Gi"
  num_accelerators: 1
"""
    with open(model_path / "config.yaml", "w") as f:
        f.write(config_content)

    # Create requirements.txt
    with open(model_path / "requirements.txt", "w") as f:
        f.write("clarifai>=10.0.0\n")

    # Create model.py
    model_content = """
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Text

class MyModel(ModelClass):
    def load_model(self):
        pass

    @ModelClass.method
    def predict(self, text1: Text = "") -> Text:
        return Text(text1.text + "Hello World")

    def test(self):
        res = self.predict(Text("test"))
        assert res.text == "testHello World"
"""
    with open(version_path / "model.py", "w") as f:
        f.write(model_content)

    return str(model_path)


@pytest.fixture
def cpu_model_path(tmp_path):
    """Create a test model that only uses CPU."""
    model_path = tmp_path / "cpu_model"
    model_path.mkdir()

    # Create version folder
    version_path = model_path / "1"
    version_path.mkdir()

    # Create config.yaml for CPU-only
    config_content = """
model:
  id: "test-cpu-model"
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
        f.write("clarifai>=10.0.0\n")

    # Create model.py
    model_content = """
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Text

class MyModel(ModelClass):
    def load_model(self):
        pass

    @ModelClass.method
    def predict(self, text1: Text = "") -> Text:
        return Text(text1.text + "Hello World")

    def test(self):
        res = self.predict(Text("test"))
        assert res.text == "testHello World"
"""
    with open(version_path / "model.py", "w") as f:
        f.write(model_content)

    return str(model_path)


class TestEnvironmentValidation:
    """Test environment validation for unsupported configurations."""

    def test_python_version_validation_current(self, gpu_model_path):
        """Test that current Python version passes validation."""
        manager = ModelRunLocally(gpu_model_path)
        # Should not raise SystemExit
        manager._validate_test_environment()

    def test_python_version_validation_old(self, gpu_model_path):
        """Test that old Python versions fail validation."""
        manager = ModelRunLocally(gpu_model_path)

        with patch('sys.version_info', (3, 7, 0)):
            with pytest.raises(SystemExit):
                manager._validate_test_environment()

    def test_macos_gpu_model_fails(self, gpu_model_path):
        """Test that macOS fails for GPU models without nvidia-smi."""
        manager = ModelRunLocally(gpu_model_path)

        with patch('platform.system', return_value='Darwin'), patch(
            'shutil.which', return_value=None
        ):  # No nvidia-smi (GPU not available)
            with pytest.raises(SystemExit):
                manager._validate_test_environment()

    def test_macos_cpu_model_warns(self, cpu_model_path):
        """Test that macOS shows warnings for CPU models."""
        manager = ModelRunLocally(cpu_model_path)

        with patch('platform.system', return_value='Darwin'), patch(
            'shutil.which', return_value=None
        ):  # No nvidia-smi or docker
            # Should pass but with warnings (no GPU required, num_accelerators=0)
            manager._validate_test_environment()

    def test_linux_validation_passes(self, gpu_model_path):
        """Test that Linux generally passes validation."""
        manager = ModelRunLocally(gpu_model_path)

        with patch('platform.system', return_value='Linux'):
            # Should pass without issues
            manager._validate_test_environment()

    def test_validation_called_in_main(self, gpu_model_path):
        """Test that validation is called in the main function."""

        with patch(
            'clarifai.runners.models.model_run_locally.ModelRunLocally._validate_test_environment'
        ) as mock_validate, patch(
            'clarifai.runners.models.model_run_locally.ModelRunLocally.create_temp_venv'
        ), patch(
            'clarifai.runners.models.model_run_locally.ModelRunLocally.install_requirements'
        ), patch('clarifai.runners.models.model_run_locally.ModelRunLocally.test_model'), patch(
            'clarifai.runners.models.model_run_locally.ModelRunLocally.clean_up'
        ):
            try:
                main(gpu_model_path, keep_env=True)
            except:
                pass  # We expect some exceptions due to mocking

            # Ensure validation was called
            assert mock_validate.called, "Environment validation should be called in main()"

    def test_validation_early_exit(self, gpu_model_path):
        """Test that validation causes early exit on failure."""

        # Mock validation to fail
        with patch(
            'clarifai.runners.models.model_run_locally.ModelRunLocally._validate_test_environment',
            side_effect=SystemExit(1),
        ), patch(
            'clarifai.runners.models.model_run_locally.ModelRunLocally.create_temp_venv'
        ) as mock_venv:
            with pytest.raises(SystemExit):
                main(gpu_model_path)

            # Ensure subsequent operations are not called
            assert not mock_venv.called, (
                "create_temp_venv should not be called if validation fails"
            )

    def test_num_accelerators_detection(self, tmp_path):
        """Test that num_accelerators field correctly determines GPU requirement."""
        # Create model with num_accelerators > 0
        model_path = tmp_path / "gpu_model_num_acc"
        model_path.mkdir()
        version_path = model_path / "1"
        version_path.mkdir()

        config_content = """
model:
  id: "test-gpu-model-num-acc"
  model_type_id: "text-to-text"
  user_id: "test_user"
  app_id: "test_app"

build_info:
  python_version: "3.12"

inference_compute_info:
  cpu_limit: "1"
  cpu_memory: "1Gi"
  num_accelerators: 2
"""
        with open(model_path / "config.yaml", "w") as f:
            f.write(config_content)

        with open(model_path / "requirements.txt", "w") as f:
            f.write("clarifai>=10.0.0\n")

        model_content = """
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Text

class MyModel(ModelClass):
    def load_model(self):
        pass

    @ModelClass.method
    def predict(self, text1: Text = "") -> Text:
        return Text(text1.text + "Hello World")

    def test(self):
        res = self.predict(Text("test"))
        assert res.text == "testHello World"
"""
        with open(version_path / "model.py", "w") as f:
            f.write(model_content)

        manager = ModelRunLocally(str(model_path))

        # Test on environment without GPU - should fail
        with patch('platform.system', return_value='Darwin'), patch(
            'shutil.which', return_value=None
        ):  # No nvidia-smi
            with pytest.raises(SystemExit):
                manager._validate_test_environment()

    def test_zero_accelerators_cpu_only(self, cpu_model_path):
        """Test that num_accelerators=0 is treated as CPU-only and passes with warnings."""
        manager = ModelRunLocally(cpu_model_path)

        # Test on environment without GPU - should pass with warnings
        with patch('platform.system', return_value='Darwin'), patch(
            'shutil.which', return_value=None
        ):  # No nvidia-smi or docker
            # Should pass (CPU-only model) but with warnings about no GPU
            manager._validate_test_environment()
