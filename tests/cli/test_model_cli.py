import os
import tempfile
from types import SimpleNamespace

import pytest
import yaml

from clarifai.cli.model import ensure_config_exists_for_upload
from clarifai.utils.cli import customize_ollama_model
from clarifai.utils.config import Config


def test_ensure_config_exists_for_upload_creates_file(monkeypatch, tmp_path):
    model_dir = tmp_path / "my_model"
    model_dir.mkdir()
    (model_dir / "1").mkdir()
    (model_dir / "1" / "model.py").write_text("class Dummy:\n    pass\n")
    (model_dir / "requirements.txt").write_text("clarifai==10.0.0\n")

    responses = iter(
        [
            "",  # context selection (keep current)
            "custom-model",  # model id
            "",  # user id (default)
            "",  # app id (default)
            "",  # model type id
            "",  # python version
            "",  # cpu limit
            "",  # cpu memory limit
            "",  # cpu requests
            "",  # cpu memory requests
            "",  # number of accelerators
            "",  # accelerator types
            "",  # accelerator memory
            "n",  # checkpoints
            "n",  # num_threads
        ]
    )

    def fake_input(prompt=""):
        try:
            return next(responses)
        except StopIteration:
            pytest.fail(f"Unexpected prompt: {prompt!r}")

    monkeypatch.setattr("builtins.input", fake_input)

    monkeypatch.delenv("CLARIFAI_USER_ID", raising=False)
    monkeypatch.delenv("CLARIFAI_APP_ID", raising=False)
    monkeypatch.delenv("CLARIFAI_PAT", raising=False)
    monkeypatch.delenv("CLARIFAI_API_BASE", raising=False)

    ctx_config = Config(
        current_context="test-context",
        filename=str(tmp_path / "ctx.yaml"),
        contexts={
            "test-context": {
                "name": "test-context",
                "env": {
                    "CLARIFAI_USER_ID": "user-123",
                    "CLARIFAI_APP_ID": "app-456",
                    "CLARIFAI_PAT": "pat",
                    "CLARIFAI_API_BASE": "https://api",
                },
            }
        },
    )

    ctx = SimpleNamespace(obj=ctx_config)

    ensure_config_exists_for_upload(ctx, str(model_dir))

    config_path = model_dir / "config.yaml"
    assert config_path.exists()

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["id"] == "custom-model"
    assert config["model"]["user_id"] == "user-123"
    assert config["model"]["app_id"] == "app-456"
    assert config["model"]["model_type_id"] == "any-to-any"
    assert config["build_info"]["python_version"] == "3.12"
    assert config["inference_compute_info"]["num_accelerators"] == 1
    assert config["inference_compute_info"]["accelerator_type"] == ["NVIDIA-*"]
    assert config["inference_compute_info"]["cpu_limit"] == "1"
    assert config["inference_compute_info"]["cpu_memory"] == "2Gi"
    assert config["inference_compute_info"]["cpu_requests"] == "1"
    assert config["inference_compute_info"]["cpu_memory_requests"] == "1Gi"
    assert config["inference_compute_info"]["accelerator_memory"] == "15Gi"
    assert "checkpoints" not in config
    assert "num_threads" not in config


class TestModelCliOllama:
    """Test CLI model commands with Ollama toolkit integration."""

    def test_customize_ollama_model_function_call(self):
        """Test that customize_ollama_model is called with correct parameters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a mock config.yaml file
            config_file = os.path.join(tmp_dir, 'config.yaml')
            config_data = {
                'model': {'id': 'test-model', 'user_id': 'old-user-id'},
                'toolkit': None,
            }

            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f)

            # Test the function directly
            test_user_id = 'test-user-123'
            customize_ollama_model(model_path=tmp_dir, user_id=test_user_id, verbose=True)

            # Verify the config was updated
            with open(config_file, 'r', encoding='utf-8') as f:
                updated_config = yaml.safe_load(f)

            assert updated_config['model']['user_id'] == test_user_id
            assert 'toolkit' in updated_config

    def test_customize_ollama_model_missing_user_id_raises_error(self):
        """Test that customize_ollama_model raises TypeError when user_id is missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = os.path.join(tmp_dir, 'config.yaml')
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump({'model': {'id': 'test'}}, f)

            # This should raise TypeError due to missing required user_id parameter
            with pytest.raises(
                TypeError, match="missing 1 required positional argument: 'user_id'"
            ):
                getattr(customize_ollama_model, "__call__")(model_path=tmp_dir)

    def test_customize_ollama_model_with_all_parameters(self):
        """Test customize_ollama_model with all optional parameters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create config.yaml
            config_file = os.path.join(tmp_dir, 'config.yaml')
            config_data = {'model': {'id': 'test-model', 'user_id': 'old-user'}, 'toolkit': None}

            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f)

            # Create model.py template file in the correct subdirectory
            model_dir = os.path.join(tmp_dir, '1')
            os.makedirs(model_dir, exist_ok=True)
            model_file = os.path.join(model_dir, 'model.py')
            with open(model_file, 'w', encoding='utf-8') as f:
                f.write('''
# Template model file
import os

class ModelClass:
    def __init__(self):
        self.model = os.environ.get("OLLAMA_MODEL_NAME", 'llama3.2')

PORT = '23333'
context_length = '8192'
''')

            # Call with all parameters
            customize_ollama_model(
                model_path=tmp_dir,
                user_id='new-user-id',
                model_name='mistral',
                port='8080',
                context_length='8192',
                verbose=True,
            )

            # Verify config.yaml was updated
            with open(config_file, 'r', encoding='utf-8') as f:
                updated_config = yaml.safe_load(f)

            assert updated_config['model']['user_id'] == 'new-user-id'
            assert updated_config['toolkit']['model'] == 'mistral'
            assert updated_config['toolkit']['port'] == '8080'
            assert updated_config['toolkit']['context_length'] == '8192'

            # Verify model.py was updated
            with open(model_file, 'r', encoding='utf-8') as f:
                updated_model_content = f.read()

            assert (
                'self.model = os.environ.get("OLLAMA_MODEL_NAME", \'mistral\')'
                in updated_model_content
            )
            assert 'PORT = \'8080\'' in updated_model_content
            assert 'context_length = \'8192\'' in updated_model_content
