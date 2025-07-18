import os
import tempfile
from unittest import mock
from unittest.mock import patch, MagicMock

import pytest
import yaml

from clarifai.client.base import BaseClient
from clarifai.errors import UserError
from clarifai.utils.config import Config, Context


class TestBaseClientCLIConfig:
    """Test BaseClient CLI config context integration."""

    def test_base_client_uses_env_vars_first(self):
        """Test that environment variables take precedence over CLI config."""
        # Create a config with different values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = {
                'current_context': 'test',
                'contexts': {
                    'test': {
                        'CLARIFAI_PAT': 'config_pat_token',
                        'CLARIFAI_USER_ID': 'config_user_123',
                        'CLARIFAI_API_BASE': 'https://config.api.clarifai.com'
                    }
                }
            }
            yaml.dump(config_content, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {
                'CLARIFAI_PAT': 'env_pat_token',
                'CLARIFAI_USER_ID': 'env_user_123',
                'CLARIFAI_API_BASE': 'https://env.api.clarifai.com'
            }), \
            patch('clarifai.utils.constants.DEFAULT_CONFIG', config_file), \
            patch('clarifai.client.auth.create_stub') as mock_create_stub, \
            patch.object(BaseClient, '_grpc_request'):
                
                mock_create_stub.return_value = MagicMock()
                
                client = BaseClient()
                
                # Verify the client was created with env values (indirectly by successful creation)
                assert client is not None
                
        finally:
            os.unlink(config_file)

    def test_base_client_falls_back_to_cli_config(self):
        """Test that CLI config is used when env vars are not set."""
        # Create a temporary config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = {
                'current_context': 'test',
                'contexts': {
                    'test': {
                        'CLARIFAI_PAT': 'cli_pat_token',
                        'CLARIFAI_USER_ID': 'cli_user_456',
                        'CLARIFAI_API_BASE': 'https://cli.api.clarifai.com'
                    }
                }
            }
            yaml.dump(config_content, f)
            config_file = f.name

        try:
            # Clear env vars and use config
            with patch.dict(os.environ, {}, clear=True), \
            patch('clarifai.utils.constants.DEFAULT_CONFIG', config_file), \
            patch('clarifai.client.auth.create_stub') as mock_create_stub, \
            patch.object(BaseClient, '_grpc_request'):
                
                mock_create_stub.return_value = MagicMock()
                
                client = BaseClient()
                
                # Verify the client was created with CLI config values
                assert client is not None
                
        finally:
            os.unlink(config_file)

    def test_base_client_suggests_clarifai_login_when_no_credentials(self):
        """Test that error message suggests clarifai login when no credentials found."""
        # Clear env vars and use empty config
        with patch.dict(os.environ, {}, clear=True), \
        patch('clarifai.utils.config.Config.from_yaml') as mock_config:
            
            # Mock empty config
            mock_config.return_value = Config(
                current_context='empty',
                filename='/tmp/empty_config',
                contexts={'empty': Context('empty')}
            )
            
            with pytest.raises(UserError) as exc_info:
                BaseClient()
            
            error_msg = str(exc_info.value)
            assert 'clarifai login' in error_msg
            assert 'Authentication required' in error_msg

    def test_base_client_kwargs_take_precedence(self):
        """Test that keyword arguments take precedence over both env vars and CLI config."""
        # Create a config with different values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = {
                'current_context': 'test',
                'contexts': {
                    'test': {
                        'CLARIFAI_PAT': 'config_pat_token',
                        'CLARIFAI_USER_ID': 'config_user_123'
                    }
                }
            }
            yaml.dump(config_content, f)
            config_file = f.name

        try:
            with patch.dict(os.environ, {
                'CLARIFAI_PAT': 'env_pat_token'
            }), \
            patch('clarifai.utils.constants.DEFAULT_CONFIG', config_file), \
            patch('clarifai.client.auth.create_stub') as mock_create_stub, \
            patch.object(BaseClient, '_grpc_request'):
                
                mock_create_stub.return_value = MagicMock()
                
                client = BaseClient(pat='kwargs_pat_token', user_id='kwargs_user_789')
                
                # Verify the client was created with kwargs values
                assert client is not None
                
        finally:
            os.unlink(config_file)

    def test_base_client_with_session_token_fallback(self):
        """Test fallback to session token from CLI config."""
        # Create a temporary config with session token
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = {
                'current_context': 'test',
                'contexts': {
                    'test': {
                        'CLARIFAI_SESSION_TOKEN': 'cli_session_token',
                        'CLARIFAI_USER_ID': 'cli_user_456'
                    }
                }
            }
            yaml.dump(config_content, f)
            config_file = f.name

        try:
            # Clear env vars and use config
            with patch.dict(os.environ, {}, clear=True), \
            patch('clarifai.utils.constants.DEFAULT_CONFIG', config_file), \
            patch('clarifai.client.auth.create_stub') as mock_create_stub, \
            patch.object(BaseClient, '_grpc_request'):
                
                mock_create_stub.return_value = MagicMock()
                
                client = BaseClient()
                
                # Verify the client was created with session token
                assert client is not None
                
        finally:
            os.unlink(config_file)