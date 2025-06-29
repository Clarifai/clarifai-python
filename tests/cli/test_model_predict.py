"""Tests for the updated CLI model predict command with pythonic model support."""

import json
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from clarifai.cli.model import predict, _validate_inputs_against_signature, _validate_parameter_type


class TestModelPredictCLI:
    """Test the updated model predict CLI command."""

    def test_help_message_contains_new_options(self):
        """Test that help message contains new pythonic model options."""
        runner = CliRunner()
        result = runner.invoke(predict, ['--help'])
        
        assert result.exit_code == 0
        assert '--model_path' in result.output
        assert '--method' in result.output
        assert '--inputs' in result.output
        assert 'pythonic models' in result.output
        assert 'traditional models' in result.output

    def test_input_validation_valid_inputs(self):
        """Test input validation with valid inputs."""
        model_path = '/tmp/test_model'
        inputs = {'prompt': 'Hello world', 'max_tokens': 100}
        
        # Should not raise an exception
        _validate_inputs_against_signature(model_path, 'predict', inputs)

    def test_input_validation_invalid_method(self):
        """Test input validation with invalid method."""
        model_path = '/tmp/test_model'
        inputs = {'prompt': 'Hello world'}
        
        with pytest.raises(ValueError, match="Method 'invalid' not found"):
            _validate_inputs_against_signature(model_path, 'invalid', inputs)

    def test_input_validation_missing_required_params(self):
        """Test input validation with missing required parameters."""
        model_path = '/tmp/test_model'
        inputs = {'max_tokens': 100}  # Missing required 'prompt'
        
        with pytest.raises(ValueError, match="Missing required parameters"):
            _validate_inputs_against_signature(model_path, 'predict', inputs)

    def test_parameter_type_validation_valid_types(self):
        """Test parameter type validation with valid types."""
        class MockField:
            def __init__(self, name, field_type):
                self.name = name
                self.type = field_type
        
        # These should not raise exceptions
        _validate_parameter_type(MockField('prompt', 'STR'), 'hello')
        _validate_parameter_type(MockField('tokens', 'INT'), 100)
        _validate_parameter_type(MockField('temp', 'FLOAT'), 0.8)
        _validate_parameter_type(MockField('temp', 'FLOAT'), 1)  # int should work for float
        _validate_parameter_type(MockField('list', 'LIST'), [1, 2, 3])

    def test_parameter_type_validation_invalid_types(self):
        """Test parameter type validation with invalid types."""
        class MockField:
            def __init__(self, name, field_type):
                self.name = name
                self.type = field_type
        
        with pytest.raises(ValueError, match="expects string"):
            _validate_parameter_type(MockField('prompt', 'STR'), 123)
        
        with pytest.raises(ValueError, match="expects integer"):
            _validate_parameter_type(MockField('tokens', 'INT'), "hello")
        
        with pytest.raises(ValueError, match="expects float"):
            _validate_parameter_type(MockField('temp', 'FLOAT'), "hello")
        
        with pytest.raises(ValueError, match="expects list"):
            _validate_parameter_type(MockField('list', 'LIST'), "hello")

    @patch('clarifai.cli.model.validate_context')
    @patch('clarifai.client.model.Model')
    def test_predict_command_pythonic_model(self, mock_model_class, mock_validate):
        """Test predict command with pythonic model parameters."""
        runner = CliRunner()
        
        # Mock the model
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Create a mock context
        mock_context = MagicMock()
        mock_context.obj = {'pat': 'test_pat', 'base_url': 'test_url'}
        
        result = runner.invoke(predict, [
            '--model_id', 'test_model',
            '--user_id', 'test_user',
            '--app_id', 'test_app',
            '--method', 'predict',
            '--inputs', '{"prompt": "Hello world"}'
        ], obj=mock_context)
        
        assert result.exit_code == 0
        assert 'Pythonic model prediction:' in result.output
        assert 'Method: predict' in result.output
        assert '"prompt": "Hello world"' in result.output

    @patch('clarifai.cli.model.validate_context')
    @patch('clarifai.client.model.Model')
    def test_predict_command_pythonic_model_with_validation(self, mock_model_class, mock_validate):
        """Test predict command with pythonic model parameters and model path validation."""
        runner = CliRunner()
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        mock_context = MagicMock()
        mock_context.obj = {'pat': 'test_pat', 'base_url': 'test_url'}
        
        result = runner.invoke(predict, [
            '--model_id', 'test_model',
            '--user_id', 'test_user',
            '--app_id', 'test_app',
            '--model_path', '/tmp/test_model',
            '--method', 'predict',
            '--inputs', '{"prompt": "Hello world", "max_tokens": 100}'
        ], obj=mock_context)
        
        assert result.exit_code == 0
        assert 'Input validation passed' in result.output or 'Pythonic model prediction:' in result.output

    @patch('clarifai.cli.model.validate_context')
    @patch('clarifai.client.model.Model')
    def test_predict_command_invalid_json(self, mock_model_class, mock_validate):
        """Test predict command with invalid JSON input."""
        runner = CliRunner()
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        mock_context = MagicMock()
        mock_context.obj = {'pat': 'test_pat', 'base_url': 'test_url'}
        
        result = runner.invoke(predict, [
            '--model_id', 'test_model',
            '--user_id', 'test_user',
            '--app_id', 'test_app',
            '--inputs', '{"prompt": "Hello", invalid json'
        ], obj=mock_context)
        
        assert result.exit_code != 0
        assert 'Invalid JSON' in str(result.exception)

    @patch('clarifai.cli.model.validate_context')
    @patch('clarifai.client.model.Model')
    def test_predict_command_traditional_model(self, mock_model_class, mock_validate):
        """Test predict command with traditional model parameters (backward compatibility)."""
        runner = CliRunner()
        
        mock_model = MagicMock()
        mock_model.predict_by_bytes.return_value = "Mocked prediction result"
        mock_model_class.return_value = mock_model
        
        mock_context = MagicMock()
        mock_context.obj = {'pat': 'test_pat', 'base_url': 'test_url'}
        
        result = runner.invoke(predict, [
            '--model_id', 'test_model',
            '--user_id', 'test_user',
            '--app_id', 'test_app',
            '--bytes', 'Hello world'
        ], obj=mock_context)
        
        assert result.exit_code == 0
        mock_model.predict_by_bytes.assert_called_once()
        assert 'Mocked prediction result' in result.output

    @patch('clarifai.cli.model.validate_context')
    @patch('clarifai.client.model.Model')
    def test_predict_command_traditional_model_file_path(self, mock_model_class, mock_validate):
        """Test predict command with traditional model file path."""
        runner = CliRunner()
        
        mock_model = MagicMock()
        mock_model.predict_by_filepath.return_value = "File prediction result"
        mock_model_class.return_value = mock_model
        
        mock_context = MagicMock()
        mock_context.obj = {'pat': 'test_pat', 'base_url': 'test_url'}
        
        result = runner.invoke(predict, [
            '--model_id', 'test_model',
            '--user_id', 'test_user',
            '--app_id', 'test_app',
            '--file_path', '/tmp/test.txt'
        ], obj=mock_context)
        
        assert result.exit_code == 0
        mock_model.predict_by_filepath.assert_called_once()

    @patch('clarifai.cli.model.validate_context')
    @patch('clarifai.client.model.Model')
    def test_predict_command_traditional_model_url(self, mock_model_class, mock_validate):
        """Test predict command with traditional model URL."""
        runner = CliRunner()
        
        mock_model = MagicMock()
        mock_model.predict_by_url.return_value = "URL prediction result"
        mock_model_class.return_value = mock_model
        
        mock_context = MagicMock()
        mock_context.obj = {'pat': 'test_pat', 'base_url': 'test_url'}
        
        result = runner.invoke(predict, [
            '--model_id', 'test_model',
            '--user_id', 'test_user',
            '--app_id', 'test_app',
            '--url', 'https://example.com/test.jpg'
        ], obj=mock_context)
        
        assert result.exit_code == 0
        mock_model.predict_by_url.assert_called_once()

    def test_json_input_parsing(self):
        """Test JSON input parsing functionality."""
        # Valid JSON
        valid_json = '{"prompt": "Hello", "max_tokens": 100}'
        parsed = json.loads(valid_json)
        assert parsed['prompt'] == "Hello"
        assert parsed['max_tokens'] == 100
        
        # Invalid JSON should raise JSONDecodeError
        invalid_json = '{"prompt": "Hello", "max_tokens":'
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)

    def test_available_methods_in_error_message(self):
        """Test that error messages include available methods."""
        model_path = '/tmp/test_model'
        inputs = {'prompt': 'Hello world'}
        
        try:
            _validate_inputs_against_signature(model_path, 'nonexistent', inputs)
            assert False, "Should have raised an exception"
        except ValueError as e:
            assert 'Available methods:' in str(e)
            assert 'predict' in str(e)
            assert 'generate' in str(e)


if __name__ == '__main__':
    # If running directly, run a subset of tests manually
    test_instance = TestModelPredictCLI()
    
    print("Running key CLI tests...")
    
    try:
        test_instance.test_help_message_contains_new_options()
        print("✓ Help message test passed")
    except Exception as e:
        print(f"✗ Help message test failed: {e}")
    
    try:
        test_instance.test_input_validation_valid_inputs()
        print("✓ Input validation test passed")
    except Exception as e:
        print(f"✗ Input validation test failed: {e}")
    
    try:
        test_instance.test_parameter_type_validation_valid_types()
        print("✓ Parameter type validation test passed")
    except Exception as e:
        print(f"✗ Parameter type validation test failed: {e}")
    
    try:
        test_instance.test_json_input_parsing()
        print("✓ JSON parsing test passed")
    except Exception as e:
        print(f"✗ JSON parsing test failed: {e}")
    
    try:
        test_instance.test_available_methods_in_error_message()
        print("✓ Error message test passed")
    except Exception as e:
        print(f"✗ Error message test failed: {e}")
    
    print("\n🎉 Key CLI tests completed!")