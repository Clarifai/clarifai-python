"""Test file for artifact functionality."""

from unittest.mock import Mock, patch

import pytest

from clarifai.client.artifact import Artifact
from clarifai.errors import UserError


class TestArtifact:
    """Test class for Artifact client."""

    def _create_mock_artifact(self):
        """Helper method to create an Artifact with properly mocked dependencies."""
        with patch('clarifai.client.base.BaseClient.__init__', return_value=None):
            artifact = Artifact()
            # Mock the auth_helper attribute that would normally be set by BaseClient.__init__
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = Mock()
            mock_auth_helper.user_id = "mock_user"
            artifact.auth_helper = mock_auth_helper
            return artifact

    def test_init(self):
        """Test artifact initialization."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            artifact = Artifact(
                artifact_id="test_artifact", user_id="test_user", app_id="test_app"
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
                base_url="https://api.clarifai.com",
            )
            assert artifact.artifact_id == "test_artifact"

    def test_repr(self):
        """Test artifact string representation."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            artifact = Artifact(
                artifact_id="test_artifact", user_id="test_user", app_id="test_app"
            )

            repr_str = repr(artifact)
            assert "test_artifact" in repr_str
            assert "test_user" in repr_str
            assert "test_app" in repr_str

    def test_create_success(self):
        """Test successful artifact creation."""
        # Mock the response
        mock_response = Mock()
        mock_artifact = Mock()
        mock_artifact.id = "new_artifact"
        mock_artifact.user_id = "test_user"
        mock_artifact.app_id = "test_app"
        mock_response.artifacts = [mock_artifact]
        mock_response.status.code = 10000  # SUCCESS

        with (
            patch('clarifai.client.base.BaseClient.__init__', return_value=None),
            patch.object(Artifact, '_grpc_request') as mock_grpc_request,
        ):
            mock_grpc_request.return_value = mock_response

            artifact = Artifact()
            # Mock the auth_helper attribute that would normally be set by BaseClient.__init__
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = Mock()
            artifact.auth_helper = mock_auth_helper

            result = artifact.create(
                artifact_id="new_artifact", user_id="test_user", app_id="test_app"
            )

            assert isinstance(result, Artifact)
            mock_grpc_request.assert_called_once()
            # Verify the call was made with the correct method name
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == "PostArtifacts"

    def test_create_missing_params(self):
        """Test artifact creation with missing parameters."""
        with patch('clarifai.client.base.BaseClient.__init__', return_value=None):
            artifact = Artifact()
            # Mock the auth_helper attribute
            mock_auth_helper = Mock()
            artifact.auth_helper = mock_auth_helper

            with pytest.raises(UserError, match="user_id is required"):
                artifact.create(artifact_id="")

    def test_delete_success(self):
        """Test successful artifact deletion."""
        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS

        with (
            patch('clarifai.client.base.BaseClient.__init__', return_value=None),
            patch.object(Artifact, '_grpc_request') as mock_grpc_request,
        ):
            mock_grpc_request.return_value = mock_response

            artifact = Artifact()
            # Mock the auth_helper attribute
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = Mock()
            artifact.auth_helper = mock_auth_helper

            result = artifact.delete(
                artifact_id="test_artifact", user_id="test_user", app_id="test_app"
            )

            assert result is True
            mock_grpc_request.assert_called_once()
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == "DeleteArtifact"

    def test_info_success(self):
        """Test successful artifact info retrieval."""
        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS
        mock_response.artifact.id = "test_artifact"
        mock_response.artifact.user_id = "test_user"
        mock_response.artifact.app_id = "test_app"
        mock_response.artifact.created_at.ToDatetime.return_value = "2024-01-01"
        mock_response.artifact.modified_at = None
        mock_response.artifact.deleted_at = None
        mock_response.artifact.artifact_version = None

        with (
            patch('clarifai.client.base.BaseClient.__init__', return_value=None),
            patch.object(Artifact, '_grpc_request') as mock_grpc_request,
        ):
            mock_grpc_request.return_value = mock_response

            artifact = Artifact()
            # Mock the auth_helper attribute
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = Mock()
            artifact.auth_helper = mock_auth_helper

            info = artifact.info(
                artifact_id="test_artifact", user_id="test_user", app_id="test_app"
            )

            assert info["id"] == "test_artifact"
            assert info["user_id"] == "test_user"
            assert info["app_id"] == "test_app"
            mock_grpc_request.assert_called_once()
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == "GetArtifact"

    def test_list_success(self):
        """Test successful artifact listing."""
        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS
        mock_artifact = Mock()
        mock_artifact.id = "test_artifact"
        mock_response.artifacts = [mock_artifact]

        with (
            patch('clarifai.client.base.BaseClient.__init__', return_value=None),
            patch.object(Artifact, '_grpc_request') as mock_grpc_request,
            patch('clarifai.client.artifact.BaseClient') as mock_base_client_class,
        ):
            mock_grpc_request.return_value = mock_response

            # Create a mock BaseClient instance for the static method
            mock_base_client = Mock()
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = Mock()
            mock_base_client.auth_helper = mock_auth_helper
            mock_base_client._grpc_request = mock_grpc_request
            mock_base_client_class.return_value = mock_base_client

            artifacts = list(Artifact.list(user_id="test_user", app_id="test_app"))

            assert len(artifacts) == 1
            assert artifacts[0].artifact_id == "test_artifact"
            mock_grpc_request.assert_called_once()
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == "ListArtifacts"

    def test_exists_true(self):
        """Test artifact exists returns True when artifact is found."""
        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS
        mock_response.artifact.id = "test_artifact"
        mock_response.artifact.user_id = "test_user"
        mock_response.artifact.app_id = "test_app"
        mock_response.artifact.created_at = None
        mock_response.artifact.modified_at = None
        mock_response.artifact.deleted_at = None
        mock_response.artifact.artifact_version = None

        with (
            patch('clarifai.client.base.BaseClient.__init__', return_value=None),
            patch.object(Artifact, '_grpc_request') as mock_grpc_request,
        ):
            mock_grpc_request.return_value = mock_response

            artifact = Artifact()
            # Mock the auth_helper attribute
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = Mock()
            artifact.auth_helper = mock_auth_helper

            exists = artifact.exists(
                artifact_id="test_artifact", user_id="test_user", app_id="test_app"
            )

            assert exists is True

    def test_exists_false(self):
        """Test artifact exists returns False when artifact is not found."""
        with (
            patch('clarifai.client.base.BaseClient.__init__', return_value=None),
            patch.object(Artifact, '_grpc_request') as mock_grpc_request,
        ):
            mock_grpc_request.side_effect = Exception("Not found")

            artifact = Artifact()
            # Mock the auth_helper attribute
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = Mock()
            artifact.auth_helper = mock_auth_helper

            exists = artifact.exists(
                artifact_id="test_artifact", user_id="test_user", app_id="test_app"
            )

            assert exists is False


class TestArtifactValidation:
    """Test class for artifact validation."""

    def test_create_missing_user_id(self):
        """Test artifact creation with missing user_id."""
        with patch('clarifai.client.base.BaseClient.__init__', return_value=None):
            artifact = Artifact()
            # Mock the auth_helper attribute
            mock_auth_helper = Mock()
            artifact.auth_helper = mock_auth_helper

            with pytest.raises(UserError, match="user_id is required"):
                artifact.create(artifact_id="test", user_id="", app_id="test_app")

    def test_create_missing_app_id(self):
        """Test artifact creation with missing app_id."""
        with patch('clarifai.client.base.BaseClient.__init__', return_value=None):
            artifact = Artifact()
            # Mock the auth_helper attribute
            mock_auth_helper = Mock()
            artifact.auth_helper = mock_auth_helper

            with pytest.raises(UserError, match="app_id is required"):
                artifact.create(artifact_id="test", user_id="test_user", app_id="")

    def test_delete_missing_artifact_id(self):
        """Test artifact deletion with missing artifact_id."""
        with patch('clarifai.client.base.BaseClient.__init__', return_value=None):
            artifact = Artifact()
            # Mock the auth_helper attribute
            mock_auth_helper = Mock()
            artifact.auth_helper = mock_auth_helper

            with pytest.raises(UserError, match="artifact_id is required"):
                artifact.delete(artifact_id="", user_id="test_user", app_id="test_app")

    def test_info_missing_artifact_id(self):
        """Test artifact info with missing artifact_id."""
        with patch('clarifai.client.base.BaseClient.__init__', return_value=None):
            artifact = Artifact()
            # Mock the auth_helper attribute
            mock_auth_helper = Mock()
            artifact.auth_helper = mock_auth_helper

            with pytest.raises(UserError, match="artifact_id is required"):
                artifact.info(artifact_id="", user_id="test_user", app_id="test_app")
