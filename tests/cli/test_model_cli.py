import os
import tempfile

import pytest
import yaml

from clarifai.utils.cli import customize_ollama_model


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

            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)

            # Test the function directly
            test_user_id = 'test-user-123'
            customize_ollama_model(model_path=tmp_dir, user_id=test_user_id, verbose=True)

            # Verify the config was updated
            with open(config_file, 'r') as f:
                updated_config = yaml.safe_load(f)

            assert updated_config['model']['user_id'] == test_user_id
            assert 'toolkit' in updated_config

    def test_customize_ollama_model_missing_user_id_raises_error(self):
        """Test that customize_ollama_model raises TypeError when user_id is missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = os.path.join(tmp_dir, 'config.yaml')
            with open(config_file, 'w') as f:
                yaml.dump({'model': {'id': 'test'}}, f)

            # This should raise TypeError due to missing required user_id parameter
            with pytest.raises(
                TypeError, match="missing 1 required positional argument: 'user_id'"
            ):
                customize_ollama_model(model_path=tmp_dir)

    def test_customize_ollama_model_with_all_parameters(self):
        """Test customize_ollama_model with all optional parameters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create config.yaml
            config_file = os.path.join(tmp_dir, 'config.yaml')
            config_data = {'model': {'id': 'test-model', 'user_id': 'old-user'}, 'toolkit': None}

            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)

            # Create model.py template file in the correct subdirectory
            model_dir = os.path.join(tmp_dir, '1')
            os.makedirs(model_dir, exist_ok=True)
            model_file = os.path.join(model_dir, 'model.py')
            with open(model_file, 'w') as f:
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
            with open(config_file, 'r') as f:
                updated_config = yaml.safe_load(f)

            assert updated_config['model']['user_id'] == 'new-user-id'
            assert updated_config['toolkit']['model'] == 'mistral'
            assert updated_config['toolkit']['port'] == '8080'
            assert updated_config['toolkit']['context_length'] == '8192'

            # Verify model.py was updated
            with open(model_file, 'r') as f:
                updated_model_content = f.read()

            assert (
                'self.model = os.environ.get("OLLAMA_MODEL_NAME", \'mistral\')'
                in updated_model_content
            )
            assert 'PORT = \'8080\'' in updated_model_content
            assert 'context_length = \'8192\'' in updated_model_content
