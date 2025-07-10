import pytest
from unittest.mock import Mock, patch, MagicMock
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.utils.cli import validate_pat_token


def test_validate_pat_token_success():
    """Test PAT token validation with a successful API response."""
    mock_response = Mock()
    mock_response.status.code = status_code_pb2.SUCCESS
    
    with patch('clarifai.client.user.User') as mock_user_class:
        mock_user_instance = Mock()
        mock_user_instance.get_user_info.return_value = mock_response
        mock_user_class.from_auth_helper.return_value = mock_user_instance
        
        is_valid, error_message = validate_pat_token("valid_token", "test_user")
        
        assert is_valid is True
        assert error_message == ""


def test_validate_pat_token_failure_with_error_status():
    """Test PAT token validation with a failed API response."""
    mock_response = Mock()
    mock_response.status.code = status_code_pb2.FAILURE
    mock_response.status.description = "Invalid token"
    
    with patch('clarifai.client.user.User') as mock_user_class:
        mock_user_instance = Mock()
        mock_user_instance.get_user_info.return_value = mock_response
        mock_user_class.from_auth_helper.return_value = mock_user_instance
        
        is_valid, error_message = validate_pat_token("invalid_token", "test_user")
        
        assert is_valid is False
        assert "Authentication failed: Invalid token" in error_message


def test_validate_pat_token_permission_denied_exception():
    """Test PAT token validation with permission denied exception."""
    with patch('clarifai.client.user.User') as mock_user_class:
        mock_user_instance = Mock()
        mock_user_instance.get_user_info.side_effect = Exception("PERMISSION_DENIED error")
        mock_user_class.from_auth_helper.return_value = mock_user_instance
        
        is_valid, error_message = validate_pat_token("invalid_token", "test_user")
        
        assert is_valid is False
        assert "Invalid PAT token or insufficient permissions" in error_message


def test_validate_pat_token_unauthenticated_exception():
    """Test PAT token validation with unauthenticated exception."""
    with patch('clarifai.client.user.User') as mock_user_class:
        mock_user_instance = Mock()
        mock_user_instance.get_user_info.side_effect = Exception("UNAUTHENTICATED error")
        mock_user_class.from_auth_helper.return_value = mock_user_instance
        
        is_valid, error_message = validate_pat_token("invalid_token", "test_user")
        
        assert is_valid is False
        assert "Invalid PAT token" in error_message


def test_validate_pat_token_ssl_exception():
    """Test PAT token validation with SSL exception."""
    with patch('clarifai.client.user.User') as mock_user_class:
        mock_user_instance = Mock()
        mock_user_instance.get_user_info.side_effect = Exception("SSL certificate error")
        mock_user_class.from_auth_helper.return_value = mock_user_instance
        
        is_valid, error_message = validate_pat_token("invalid_token", "test_user")
        
        assert is_valid is False
        assert "SSL/Certificate error" in error_message


def test_validate_pat_token_connection_exception():
    """Test PAT token validation with connection exception."""
    with patch('clarifai.client.user.User') as mock_user_class:
        mock_user_instance = Mock()
        mock_user_instance.get_user_info.side_effect = Exception("Connection timeout")
        mock_user_class.from_auth_helper.return_value = mock_user_instance
        
        is_valid, error_message = validate_pat_token("invalid_token", "test_user")
        
        assert is_valid is False
        assert "Network connection error" in error_message


def test_validate_pat_token_generic_exception():
    """Test PAT token validation with generic exception."""
    with patch('clarifai.client.user.User') as mock_user_class:
        mock_user_instance = Mock()
        mock_user_instance.get_user_info.side_effect = Exception("Generic error")
        mock_user_class.from_auth_helper.return_value = mock_user_instance
        
        is_valid, error_message = validate_pat_token("invalid_token", "test_user")
        
        assert is_valid is False
        assert "Validation error: Generic error" in error_message


def test_validate_pat_token_with_custom_api_base():
    """Test PAT token validation with custom API base URL."""
    mock_response = Mock()
    mock_response.status.code = status_code_pb2.SUCCESS
    
    with patch('clarifai.client.auth.helper.ClarifaiAuthHelper') as mock_auth_helper, \
         patch('clarifai.client.user.User') as mock_user_class:
        
        mock_user_instance = Mock()
        mock_user_instance.get_user_info.return_value = mock_response
        mock_user_class.from_auth_helper.return_value = mock_user_instance
        
        is_valid, error_message = validate_pat_token("valid_token", "test_user", "https://custom.api.com")
        
        # Verify auth helper was called with custom base
        mock_auth_helper.assert_called_once_with(
            user_id="test_user", 
            pat="valid_token", 
            validate=False, 
            base="https://custom.api.com"
        )
        
        assert is_valid is True
        assert error_message == ""