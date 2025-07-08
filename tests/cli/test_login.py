import os
import tempfile
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
import pytest

from clarifai.cli.base import cli
from clarifai.utils.config import Context


class TestLoginCommand:
    """Test cases for the login command functionality."""

    def test_login_exports_to_environment(self):
        """Test that login command exports CLARIFAI_USER_ID and CLARIFAI_PAT to environment."""
        runner = CliRunner()
        
        # Mock the input functions to provide test data
        with patch('clarifai.cli.base.input') as mock_input, \
             patch('clarifai.cli.base.input_or_default') as mock_input_or_default, \
             patch('clarifai.utils.config.DEFAULT_CONFIG') as mock_config_path:
            
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
                mock_config_path.return_value = temp_config.name
                
                # Mock user inputs
                mock_input.side_effect = ['test_context', 'test_user_123']
                mock_input_or_default.return_value = 'test_pat_456'
                
                # Clear any existing environment variables
                env_backup = {}
                for key in ['CLARIFAI_USER_ID', 'CLARIFAI_PAT']:
                    if key in os.environ:
                        env_backup[key] = os.environ[key]
                        del os.environ[key]
                
                try:
                    # Run the login command
                    result = runner.invoke(cli, ['login'])
                    
                    # Verify the command succeeded
                    assert result.exit_code == 0
                    
                    # Check that environment variables were set
                    assert os.environ.get('CLARIFAI_USER_ID') == 'test_user_123'
                    assert os.environ.get('CLARIFAI_PAT') == 'test_pat_456'
                    
                finally:
                    # Restore original environment
                    for key in ['CLARIFAI_USER_ID', 'CLARIFAI_PAT']:
                        if key in os.environ:
                            del os.environ[key]
                    for key, value in env_backup.items():
                        os.environ[key] = value
                    
                    # Clean up temp file
                    os.unlink(temp_config.name)

    def test_context_set_to_env(self):
        """Test that Context.set_to_env() properly exports variables to environment."""
        # Create a test context
        context = Context(
            'test_context',
            CLARIFAI_USER_ID='test_user_123',
            CLARIFAI_PAT='test_pat_456',
            CLARIFAI_API_BASE='https://api.clarifai.com'
        )
        
        # Clear any existing environment variables
        env_backup = {}
        for key in ['CLARIFAI_USER_ID', 'CLARIFAI_PAT', 'CLARIFAI_API_BASE']:
            if key in os.environ:
                env_backup[key] = os.environ[key]
                del os.environ[key]
        
        try:
            # Call set_to_env
            context.set_to_env()
            
            # Check that environment variables were set
            assert os.environ.get('CLARIFAI_USER_ID') == 'test_user_123'
            assert os.environ.get('CLARIFAI_PAT') == 'test_pat_456'
            assert os.environ.get('CLARIFAI_API_BASE') == 'https://api.clarifai.com'
            
        finally:
            # Restore original environment
            for key in ['CLARIFAI_USER_ID', 'CLARIFAI_PAT', 'CLARIFAI_API_BASE']:
                if key in os.environ:
                    del os.environ[key]
            for key, value in env_backup.items():
                os.environ[key] = value