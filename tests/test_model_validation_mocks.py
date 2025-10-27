"""Unit tests for Model initialization validation with mocks."""

from unittest.mock import MagicMock, patch

import pytest
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.model import Model
from clarifai.errors import UserError


class TestModelValidationWithMocks:
    """Test Model validation using mocks."""

    @patch('clarifai.client.model.Model._validate_model_exists')
    @patch('clarifai.client.model.BaseClient.__init__', return_value=None)
    @patch('clarifai.client.model.Lister.__init__', return_value=None)
    @patch('clarifai.client.model.Model._set_runner_selector')
    def test_validation_called_by_default(
        self, mock_runner_selector, mock_lister_init, mock_base_init, mock_validate
    ):
        """Test that validation is called by default when creating a Model."""
        # Create Model with default validate=True
        model = Model(model_id='test_model', user_id='test_user', app_id='test_app')

        # Verify validation was called once
        assert mock_validate.call_count == 1

    @patch('clarifai.client.model.Model._validate_model_exists')
    @patch('clarifai.client.model.BaseClient.__init__', return_value=None)
    @patch('clarifai.client.model.Lister.__init__', return_value=None)
    @patch('clarifai.client.model.Model._set_runner_selector')
    def test_validation_skipped_when_false(
        self, mock_runner_selector, mock_lister_init, mock_base_init, mock_validate
    ):
        """Test that validation is skipped when validate=False."""
        # Create Model with validate=False
        model = Model(
            model_id='test_model', user_id='test_user', app_id='test_app', validate=False
        )

        # Verify validation was NOT called
        mock_validate.assert_not_called()

    def test_validate_model_exists_with_success(self):
        """Test _validate_model_exists when model exists."""
        model = MagicMock()
        model.user_app_id = MagicMock()
        model.id = 'test_model'
        model.model_info.model_version.id = 'version1'

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status.code = status_code_pb2.SUCCESS

        model._grpc_request = MagicMock(return_value=mock_response)

        # Call the actual method
        Model._validate_model_exists(model, original_url=None)

        # Should not raise any exception
        assert model._grpc_request.called

    def test_validate_model_exists_with_failure(self):
        """Test _validate_model_exists when model does not exist."""
        model = MagicMock()
        model.user_app_id = MagicMock()
        model.id = 'nonexistent_model'
        model.app_id = 'test_app'
        model.user_id = 'test_user'
        model.model_info.model_version.id = ''

        # Mock failure response
        mock_response = MagicMock()
        mock_response.status.code = status_code_pb2.MODEL_DOES_NOT_EXIST
        mock_response.status.description = 'Model not found'
        mock_response.status.details = 'The requested model does not exist'

        model._grpc_request = MagicMock(return_value=mock_response)

        # Should raise UserError
        with pytest.raises(UserError) as exc_info:
            Model._validate_model_exists(model, original_url=None)

        error_msg = str(exc_info.value)
        assert 'does not exist' in error_msg or 'cannot be accessed' in error_msg
        assert 'nonexistent_model' in error_msg

    def test_validate_model_exists_with_url(self):
        """Test _validate_model_exists error message includes URL when provided."""
        model = MagicMock()
        model.user_app_id = MagicMock()
        model.id = 'test_model'
        model.model_info.model_version.id = ''

        # Mock failure response
        mock_response = MagicMock()
        mock_response.status.code = status_code_pb2.MODEL_DOES_NOT_EXIST
        mock_response.status.description = 'Model not found'
        mock_response.status.details = 'The requested model does not exist'

        model._grpc_request = MagicMock(return_value=mock_response)

        test_url = 'https://clarifai.com/test_user/test_app/models/test_model'

        # Should raise UserError with URL in message
        with pytest.raises(UserError) as exc_info:
            Model._validate_model_exists(model, original_url=test_url)

        error_msg = str(exc_info.value)
        assert test_url in error_msg
