"""Test file for artifact functionality."""

from unittest.mock import Mock, patch

import pytest
from clarifai_grpc.grpc.api import resources_pb2

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
            mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
                user_id="test_user", app_id="test_app"
            )
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
            # Mock the auth_helper and STUB attributes that would normally be set by BaseClient.__init__
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
                user_id="test_user", app_id="test_app"
            )
            artifact.auth_helper = mock_auth_helper
            mock_stub = Mock()
            mock_stub.PostArtifacts = Mock()
            artifact.STUB = mock_stub

            result = artifact.create(
                artifact_id="new_artifact", user_id="test_user", app_id="test_app"
            )

            assert isinstance(result, Artifact)
            mock_grpc_request.assert_called_once()
            # Verify the call was made with the correct method object
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == mock_stub.PostArtifacts

    def test_create_missing_params(self):
        """Test artifact creation with missing parameters."""
        with patch('clarifai.client.base.BaseClient.__init__', return_value=None):
            artifact = Artifact()
            # Mock the auth_helper attribute
            mock_auth_helper = Mock()
            artifact.auth_helper = mock_auth_helper

            with pytest.raises(UserError, match="artifact_id is required"):
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
            # Mock the auth_helper and STUB attributes
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
                user_id="test_user", app_id="test_app"
            )
            artifact.auth_helper = mock_auth_helper
            mock_stub = Mock()
            mock_stub.DeleteArtifact = Mock()
            artifact.STUB = mock_stub

            result = artifact.delete(
                artifact_id="test_artifact", user_id="test_user", app_id="test_app"
            )

            assert result is True
            mock_grpc_request.assert_called_once()
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == mock_stub.DeleteArtifact

    def test_get_success(self):
        """Test successful artifact get retrieval."""
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
            # Mock the auth_helper and STUB attributes
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
                user_id="test_user", app_id="test_app"
            )
            artifact.auth_helper = mock_auth_helper
            mock_stub = Mock()
            mock_stub.GetArtifact = Mock()
            artifact.STUB = mock_stub

            result = artifact.get(
                artifact_id="test_artifact", user_id="test_user", app_id="test_app"
            )

            assert result.id == "test_artifact"
            assert result.user_id == "test_user"
            assert result.app_id == "test_app"
            mock_grpc_request.assert_called_once()
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == mock_stub.GetArtifact

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
        ):
            mock_grpc_request.return_value = mock_response

            artifact = Artifact(user_id="test_user", app_id="test_app")
            # Mock the auth_helper and STUB attributes
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
                user_id="test_user", app_id="test_app"
            )
            artifact.auth_helper = mock_auth_helper
            mock_stub = Mock()
            mock_stub.ListArtifacts = Mock()
            artifact.STUB = mock_stub

            artifacts = list(artifact.list())

            assert len(artifacts) == 1
            assert artifacts[0].id == "test_artifact"
            mock_grpc_request.assert_called_once()
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == mock_stub.ListArtifacts

    def test_create_missing_artifact_id(self):
        """Test artifact creation with missing artifact_id (now required)."""
        artifact = self._create_mock_artifact()

        # Test missing artifact_id
        with pytest.raises(UserError, match="artifact_id is required"):
            artifact.create(
                artifact_id="",  # Empty ID should trigger error
                user_id="test_user",
                app_id="test_app",
            )


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

    def test_get_missing_artifact_id(self):
        """Test artifact get with missing artifact_id."""
        with patch('clarifai.client.base.BaseClient.__init__', return_value=None):
            artifact = Artifact()
            # Mock the auth_helper attribute
            mock_auth_helper = Mock()
            artifact.auth_helper = mock_auth_helper

            with pytest.raises(UserError, match="artifact_id is required"):
                artifact.get(artifact_id="", user_id="test_user", app_id="test_app")
