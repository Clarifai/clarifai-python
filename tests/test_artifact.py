"""Test file for artifact functionality."""

from unittest.mock import Mock, patch

import pytest
from google.protobuf import timestamp_pb2

from clarifai.client.artifact import Artifact
from clarifai.errors import UserError


class TestArtifact:
    """Test class for Artifact client."""

    def test_init(self):
        """Test artifact initialization."""
        artifact = Artifact(
            artifact_id="test_artifact",
            user_id="test_user",
            app_id="test_app"
        )

        assert artifact.artifact_id == "test_artifact"
        assert artifact.user_id == "test_user"
        assert artifact.app_id == "test_app"
        assert artifact.id == "test_artifact"

    def test_init_with_kwargs(self):
        """Test artifact initialization with kwargs."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            artifact = Artifact(
                artifact_id="test_artifact",
                user_id="test_user", 
                app_id="test_app",
                base_url="https://api.clarifai.com"
            )
            assert artifact.artifact_id == "test_artifact"

    def test_repr(self):
        """Test artifact string representation."""
        artifact = Artifact(
            artifact_id="test_artifact",
            user_id="test_user",
            app_id="test_app"
        )

        repr_str = repr(artifact)
        assert "test_artifact" in repr_str
        assert "test_user" in repr_str
        assert "test_app" in repr_str

    @patch('clarifai.client.artifact.handle_grpc_error')
    def test_create_success(self, mock_handle_grpc_error):
        """Test successful artifact creation."""
        # Mock the response
        mock_response = Mock()
        mock_artifact = Mock()
        mock_artifact.id = "new_artifact"
        mock_artifact.user_id = "test_user"
        mock_artifact.app_id = "test_app"
        mock_response.artifacts = [mock_artifact]

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(Artifact, 'V2_STUB') as mock_stub:

            mock_handle_grpc_error.return_value = mock_response

            artifact = Artifact()
            result = artifact.create(
                artifact_id="new_artifact",
                user_id="test_user",
                app_id="test_app"
            )

            assert isinstance(result, Artifact)
            mock_handle_grpc_error.assert_called_once()

    @patch('clarifai.client.artifact.handle_grpc_error')
    def test_create_with_description(self, mock_handle_grpc_error):
        """Test artifact creation with description."""
        mock_response = Mock()
        mock_artifact = Mock()
        mock_artifact.id = "new_artifact"
        mock_response.artifacts = [mock_artifact]

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(Artifact, 'V2_STUB'):

            mock_handle_grpc_error.return_value = mock_response

            artifact = Artifact()
            artifact.create(
                artifact_id="new_artifact",
                user_id="test_user", 
                app_id="test_app",
                description="Test artifact description"
            )

            # Verify the call was made with description
            call_args = mock_handle_grpc_error.call_args[0]
            assert "Test artifact description" in str(call_args)

    def test_create_missing_params(self):
        """Test artifact creation with missing parameters."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            artifact = Artifact()

            with pytest.raises(UserError, match="artifact_id is required"):
                artifact.create()

    @patch('clarifai.client.artifact.handle_grpc_error') 
    def test_delete_success(self, mock_handle_grpc_error):
        """Test successful artifact deletion."""
        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(Artifact, 'V2_STUB'):

            mock_handle_grpc_error.return_value = mock_response

            artifact = Artifact(
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app"
            )

            result = artifact.delete()
            assert result is True
            mock_handle_grpc_error.assert_called_once()

    def test_delete_missing_params(self):
        """Test artifact deletion with missing parameters."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            artifact = Artifact()

            with pytest.raises(UserError, match="artifact_id is required"):
                artifact.delete()

    @patch('clarifai.client.artifact.handle_grpc_error')
    def test_info_success(self, mock_handle_grpc_error):
        """Test successful artifact info retrieval."""
        # Create mock timestamp
        mock_timestamp = timestamp_pb2.Timestamp()
        mock_timestamp.GetCurrentTime()

        mock_response = Mock()
        mock_artifact = Mock()
        mock_artifact.id = "test_artifact"
        mock_artifact.user_id = "test_user"
        mock_artifact.app_id = "test_app"
        mock_artifact.description = "Test description"
        mock_artifact.created_at = mock_timestamp
        mock_artifact.modified_at = mock_timestamp
        mock_response.artifacts = [mock_artifact]

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(Artifact, 'V2_STUB'):

            mock_handle_grpc_error.return_value = mock_response

            artifact = Artifact(
                artifact_id="test_artifact",
                user_id="test_user", 
                app_id="test_app"
            )

            result = artifact.info()
            assert result is not None
            mock_handle_grpc_error.assert_called_once()

    @patch('clarifai.client.artifact.handle_grpc_error')
    def test_list_success(self, mock_handle_grpc_error):
        """Test successful artifact listing."""
        mock_response = Mock()
        mock_artifact1 = Mock()
        mock_artifact1.id = "artifact1"
        mock_artifact2 = Mock() 
        mock_artifact2.id = "artifact2"
        mock_response.artifacts = [mock_artifact1, mock_artifact2]

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(Artifact, 'V2_STUB'):

            mock_handle_grpc_error.return_value = mock_response

            artifact = Artifact()
            results = list(artifact.list(user_id="test_user", app_id="test_app"))

            assert len(results) == 2
            mock_handle_grpc_error.assert_called_once()

    @patch('clarifai.client.artifact.handle_grpc_error')
    def test_exists_true(self, mock_handle_grpc_error):
        """Test artifact exists method returns True."""
        mock_response = Mock()
        mock_artifact = Mock()
        mock_artifact.id = "test_artifact"
        mock_response.artifacts = [mock_artifact]

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(Artifact, 'V2_STUB'):

            mock_handle_grpc_error.return_value = mock_response

            artifact = Artifact(
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app"
            )

            result = artifact.exists()
            assert result is True

    @patch('clarifai.client.artifact.handle_grpc_error')
    def test_exists_false(self, mock_handle_grpc_error):
        """Test artifact exists method returns False."""
        mock_response = Mock()
        mock_response.artifacts = []

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(Artifact, 'V2_STUB'):

            mock_handle_grpc_error.return_value = mock_response

            artifact = Artifact(
                artifact_id="test_artifact",
                user_id="test_user", 
                app_id="test_app"
            )

            result = artifact.exists()
            assert result is False


class TestArtifactValidation:
    """Test class for artifact input validation."""

    def test_get_client_params(self):
        """Test client parameter extraction."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            artifact = Artifact(
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app"
            )

            params = artifact._get_client_params()
            expected = {
                'artifact_id': 'test_artifact',
                'user_id': 'test_user', 
                'app_id': 'test_app'
            }
            assert params == expected

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            artifact = Artifact()

            # Test various missing parameter scenarios
            with pytest.raises(UserError, match="artifact_id is required"):
                artifact.create()

            with pytest.raises(UserError, match="artifact_id is required"): 
                artifact.delete()

            with pytest.raises(UserError, match="artifact_id is required"):
                artifact.info()


if __name__ == "__main__":
    pytest.main([__file__])
