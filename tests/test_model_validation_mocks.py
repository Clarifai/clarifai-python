"""Unit tests for Model initialization validation with mocks."""

from unittest.mock import MagicMock, patch

import pytest
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.model import Model
from clarifai.errors import UserError


class TestModelValidationWithMocks:
    """Test Model validation using mocks."""

    @patch('clarifai.client.model.BaseClient.__init__')
    @patch('clarifai.client.model.Lister.__init__')
    def test_validation_called_by_default(self, mock_lister_init, mock_base_init):
        """Test that validation is called by default."""
        mock_base_init.return_value = None
        mock_lister_init.return_value = None

        # Create a mock for the Model instance
        with patch.object(Model, '_validate_model_exists') as mock_validate:
            with patch.object(Model, '_set_runner_selector'):
                with patch.object(Model, 'user_id', 'test_user'):
                    with patch.object(Model, 'app_id', 'test_app'):
                        model = Model.__new__(Model)
                        model.kwargs = {}
                        model.logger = MagicMock()
                        model.training_params = {}
                        model.input_types = None
                        model._client = None
                        model._async_client = None
                        model._added_methods = False
                        model.deployment_user_id = None

                        # Now call __init__ with validate=True (default)
                        Model.__init__(
                            model,
                            model_id='test_model',
                            user_id='test_user',
                            app_id='test_app',
                        )

                        # Verify validation was called
                        mock_validate.assert_called_once()

    @patch('clarifai.client.model.BaseClient.__init__')
    @patch('clarifai.client.model.Lister.__init__')
    def test_validation_skipped_when_false(self, mock_lister_init, mock_base_init):
        """Test that validation is skipped when validate=False."""
        mock_base_init.return_value = None
        mock_lister_init.return_value = None

        # Create a mock for the Model instance
        with patch.object(Model, '_validate_model_exists') as mock_validate:
            with patch.object(Model, '_set_runner_selector'):
                with patch.object(Model, 'user_id', 'test_user'):
                    with patch.object(Model, 'app_id', 'test_app'):
                        model = Model.__new__(Model)
                        model.kwargs = {}
                        model.logger = MagicMock()
                        model.training_params = {}
                        model.input_types = None
                        model._client = None
                        model._async_client = None
                        model._added_methods = False
                        model.deployment_user_id = None

                        # Now call __init__ with validate=False
                        Model.__init__(
                            model,
                            model_id='test_model',
                            user_id='test_user',
                            app_id='test_app',
                            validate=False,
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
