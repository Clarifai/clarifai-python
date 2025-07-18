import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from clarifai.errors import UserError
from clarifai.utils.misc import get_from_dict_env_or_config


class TestGetFromDictEnvOrConfig:
    """Test the get_from_dict_env_or_config function."""

    def test_kwargs_take_precedence(self):
        """Test that kwargs take precedence over env vars and config."""
        result = get_from_dict_env_or_config('pat', 'CLARIFAI_PAT', pat='kwargs_pat')
        assert result == 'kwargs_pat'

    def test_env_vars_take_precedence_over_config(self):
        """Test that environment variables take precedence over CLI config."""
        # Create a config with different value
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = {
                'current_context': 'test',
                'contexts': {
                    'test': {
                        'CLARIFAI_PAT': 'config_pat_token'
                    }
                }
            }
            yaml.dump(config_content, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {'CLARIFAI_PAT': 'env_pat_token'}), \
            patch('clarifai.utils.constants.DEFAULT_CONFIG', config_file):
                
                result = get_from_dict_env_or_config('pat', 'CLARIFAI_PAT')
                assert result == 'env_pat_token'
                
        finally:
            os.unlink(config_file)

    def test_falls_back_to_cli_config(self):
        """Test that CLI config is used when env vars are not set."""
        # Create a config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = {
                'current_context': 'test',
                'contexts': {
                    'test': {
                        'CLARIFAI_PAT': 'config_pat_token'
                    }
                }
            }
            yaml.dump(config_content, f)
            config_file = f.name

        try:
            # Clear env vars
            with patch.dict(os.environ, {}, clear=True), \
            patch('clarifai.utils.constants.DEFAULT_CONFIG', config_file):
                
                result = get_from_dict_env_or_config('pat', 'CLARIFAI_PAT')
                assert result == 'config_pat_token'
                
        finally:
            os.unlink(config_file)

    def test_user_id_mapping(self):
        """Test that CLARIFAI_USER_ID maps to user_id attribute."""
        # Create a config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = {
                'current_context': 'test',
                'contexts': {
                    'test': {
                        'CLARIFAI_USER_ID': 'test_user_123'
                    }
                }
            }
            yaml.dump(config_content, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {}, clear=True), \
            patch('clarifai.utils.constants.DEFAULT_CONFIG', config_file):
                
                result = get_from_dict_env_or_config('user_id', 'CLARIFAI_USER_ID')
                assert result == 'test_user_123'
                
        finally:
            os.unlink(config_file)

    def test_api_base_mapping(self):
        """Test that CLARIFAI_API_BASE maps to api_base attribute."""
        # Create a config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = {
                'current_context': 'test',
                'contexts': {
                    'test': {
                        'CLARIFAI_API_BASE': 'https://test.api.clarifai.com'
                    }
                }
            }
            yaml.dump(config_content, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {}, clear=True), \
            patch('clarifai.utils.constants.DEFAULT_CONFIG', config_file):
                
                result = get_from_dict_env_or_config('base', 'CLARIFAI_API_BASE')
                assert result == 'https://test.api.clarifai.com'
                
        finally:
            os.unlink(config_file)

    def test_session_token_fallback(self):
        """Test fallback with session token."""
        # Create a config with session token
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = {
                'current_context': 'test',
                'contexts': {
                    'test': {
                        'CLARIFAI_SESSION_TOKEN': 'test_session_token'
                    }
                }
            }
            yaml.dump(config_content, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {}, clear=True), \
            patch('clarifai.utils.constants.DEFAULT_CONFIG', config_file):
                
                result = get_from_dict_env_or_config('token', 'CLARIFAI_SESSION_TOKEN')
                assert result == 'test_session_token'
                
        finally:
            os.unlink(config_file)

    def test_raises_error_when_not_found(self):
        """Test that error suggests clarifai login when no credentials found."""
        with patch.dict(os.environ, {}, clear=True), \
        patch('clarifai.utils.config.Config.from_yaml') as mock_config:
            
            # Mock empty config that loads successfully but has no values
            from clarifai.utils.config import Config, Context
            mock_config.return_value = Config(
                current_context='empty',
                filename='/tmp/empty',
                contexts={'empty': Context('empty')}
            )
            
            with pytest.raises(UserError) as exc_info:
                get_from_dict_env_or_config('pat', 'CLARIFAI_PAT')
            
            error_msg = str(exc_info.value)
            assert 'clarifai login' in error_msg
            assert 'CLARIFAI_PAT' in error_msg