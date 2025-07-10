import pytest
from unittest.mock import patch, Mock, MagicMock
from click.testing import CliRunner

from clarifai.cli.base import login, create as context_create


class TestLoginValidation:
    """Test suite for PAT token validation during login."""

    def test_login_with_valid_pat_token(self):
        """Test login command with valid PAT token."""
        runner = CliRunner()
        
        with patch('clarifai.utils.cli.validate_pat_token') as mock_validate, \
             patch('builtins.input') as mock_input, \
             patch('clarifai.cli.base.input_or_default') as mock_input_default:
            
            # Mock user inputs
            mock_input.side_effect = ['test_context', 'test_user']
            mock_input_default.return_value = 'test_pat_token'
            
            # Mock successful validation
            mock_validate.return_value = (True, "")
            
            # Mock context object with minimal functionality
            mock_context = Mock()
            mock_context.obj = Mock()
            mock_context.obj.contexts = {}
            mock_context.obj.to_yaml = Mock()
            
            with patch('clarifai.utils.config.Context') as mock_context_class:
                mock_context_instance = Mock()
                mock_context_instance.name = 'test_context'
                mock_context_class.return_value = mock_context_instance
                
                result = runner.invoke(login, ['--user_id', 'test_user'], obj=mock_context.obj)
                
                # Verify validation was called
                mock_validate.assert_called_once_with('test_pat_token', 'test_user', 'https://api.clarifai.com')
                
                # Check that command completed successfully
                assert result.exit_code == 0
                assert "PAT token is valid" in result.output
                assert "Configuration saved successfully" in result.output

    def test_login_with_invalid_pat_token(self):
        """Test login command with invalid PAT token."""
        runner = CliRunner()
        
        with patch('clarifai.utils.cli.validate_pat_token') as mock_validate, \
             patch('builtins.input') as mock_input, \
             patch('clarifai.cli.base.input_or_default') as mock_input_default:
            
            # Mock user inputs
            mock_input.side_effect = ['test_context', 'test_user']
            mock_input_default.return_value = 'invalid_pat_token'
            
            # Mock failed validation
            mock_validate.return_value = (False, "Invalid PAT token")
            
            # Mock context object
            mock_context = Mock()
            mock_context.obj = Mock()
            mock_context.obj.contexts = {}
            
            result = runner.invoke(login, ['--user_id', 'test_user'], obj=mock_context.obj)
            
            # Verify validation was called
            mock_validate.assert_called_once_with('invalid_pat_token', 'test_user', 'https://api.clarifai.com')
            
            # Check that command exits early with error message
            assert result.exit_code == 0  # Click doesn't return non-zero for early returns
            assert "PAT token validation failed" in result.output
            assert "Invalid PAT token" in result.output
            assert "Please check your token and try again" in result.output

    def test_login_with_envvar_pat_skips_validation(self):
        """Test login command with ENVVAR PAT skips validation."""
        runner = CliRunner()
        
        with patch('clarifai.utils.cli.validate_pat_token') as mock_validate, \
             patch('builtins.input') as mock_input, \
             patch('clarifai.cli.base.input_or_default') as mock_input_default:
            
            # Mock user inputs
            mock_input.side_effect = ['test_context', 'test_user']
            mock_input_default.return_value = 'ENVVAR'
            
            # Mock context object
            mock_context = Mock()
            mock_context.obj = Mock()
            mock_context.obj.contexts = {}
            mock_context.obj.to_yaml = Mock()
            
            with patch('clarifai.utils.config.Context') as mock_context_class:
                mock_context_instance = Mock()
                mock_context_instance.name = 'test_context'
                mock_context_class.return_value = mock_context_instance
                
                result = runner.invoke(login, ['--user_id', 'test_user'], obj=mock_context.obj)
                
                # Verify validation was NOT called
                mock_validate.assert_not_called()
                
                # Check that command completed successfully
                assert result.exit_code == 0
                assert "Configuration saved successfully" in result.output

    def test_context_create_with_valid_pat_token(self):
        """Test context create command with valid PAT token."""
        runner = CliRunner()
        
        with patch('clarifai.utils.cli.validate_pat_token') as mock_validate, \
             patch('builtins.input') as mock_input, \
             patch('clarifai.cli.base.input_or_default') as mock_input_default:
            
            # Mock user inputs
            mock_input.return_value = 'test_user'
            mock_input_default.side_effect = ['https://api.clarifai.com', 'test_pat_token']
            
            # Mock successful validation
            mock_validate.return_value = (True, "")
            
            # Mock context object
            mock_context = Mock()
            mock_context.obj = Mock()
            mock_context.obj.contexts = {}
            mock_context.obj.to_yaml = Mock()
            
            with patch('clarifai.utils.config.Context') as mock_context_class:
                mock_context_instance = Mock()
                mock_context_instance.name = 'test_context'
                mock_context_class.return_value = mock_context_instance
                
                result = runner.invoke(context_create, ['test_context'], obj=mock_context.obj)
                
                # Verify validation was called
                mock_validate.assert_called_once_with('test_pat_token', 'test_user', 'https://api.clarifai.com')
                
                # Check that command completed successfully
                assert result.exit_code == 0
                assert "PAT token is valid" in result.output
                assert "Context 'test_context' created successfully" in result.output

    def test_context_create_with_invalid_pat_token(self):
        """Test context create command with invalid PAT token."""
        runner = CliRunner()
        
        with patch('clarifai.utils.cli.validate_pat_token') as mock_validate, \
             patch('builtins.input') as mock_input, \
             patch('clarifai.cli.base.input_or_default') as mock_input_default:
            
            # Mock user inputs
            mock_input.return_value = 'test_user'
            mock_input_default.side_effect = ['https://api.clarifai.com', 'invalid_pat_token']
            
            # Mock failed validation
            mock_validate.return_value = (False, "Invalid PAT token")
            
            # Mock context object
            mock_context = Mock()
            mock_context.obj = Mock()
            mock_context.obj.contexts = {}
            
            result = runner.invoke(context_create, ['test_context'], obj=mock_context.obj)
            
            # Verify validation was called
            mock_validate.assert_called_once_with('invalid_pat_token', 'test_user', 'https://api.clarifai.com')
            
            # Check that command exits early with error message
            assert result.exit_code == 0  # Click doesn't return non-zero for early returns
            assert "PAT token validation failed" in result.output
            assert "Invalid PAT token" in result.output
            assert "Please check your token and try again" in result.output

    def test_context_create_with_existing_context_name(self):
        """Test context create command with existing context name."""
        runner = CliRunner()
        
        # Mock context object with existing context
        mock_context = Mock()
        mock_context.obj = Mock()
        mock_context.obj.contexts = {'test_context': Mock()}
        
        result = runner.invoke(context_create, ['test_context'], obj=mock_context.obj)
        
        # Check that command exits with error message
        assert result.exit_code == 1
        assert "test_context already exists" in result.output