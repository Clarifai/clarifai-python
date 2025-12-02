"""Test file for artifact version functionality."""

import os
from unittest.mock import Mock, mock_open, patch

import pytest
from google.protobuf import timestamp_pb2

from clarifai.client.artifact_version import ArtifactVersion, format_bytes
from clarifai.errors import UserError


class TestArtifactVersion:
    """Test class for ArtifactVersion client."""

    def test_init(self):
        """Test artifact version initialization."""
        version = ArtifactVersion(
            artifact_id="test_artifact",
            version_id="test_version",
            user_id="test_user",
            app_id="test_app"
        )

        assert version.artifact_id == "test_artifact"
        assert version.version_id == "test_version"
        assert version.user_id == "test_user"
        assert version.app_id == "test_app"
        assert version.id == "test_version"

    def test_init_with_kwargs(self):
        """Test artifact version initialization with kwargs."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion(
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app",
                base_url="https://api.clarifai.com"
            )
            assert version.artifact_id == "test_artifact"

    def test_repr(self):
        """Test artifact version string representation."""
        version = ArtifactVersion(
            artifact_id="test_artifact",
            version_id="test_version",
            user_id="test_user",
            app_id="test_app"
        )

        repr_str = repr(version)
        assert "test_artifact" in repr_str
        assert "test_version" in repr_str
        assert "test_user" in repr_str
        assert "test_app" in repr_str

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('builtins.open', new_callable=mock_open, read_data=b'test content')
    def test_create_success(self, mock_file, mock_getsize, mock_exists):
        """Test successful artifact version creation."""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024

        # Mock streaming response
        mock_response = Mock()
        mock_response.artifact_version_id = "new_version"
        mock_response.status.code = 10000  # SUCCESS

        with (
            patch('clarifai.client.base.BaseClient.__init__'),
            patch.object(ArtifactVersion, '_grpc_request_stream') as mock_grpc_stream,
        ):
            mock_grpc_stream.return_value = [mock_response]

            version = ArtifactVersion()
            result = version.create(
                file_path="test_file.txt",
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app"
            )

            assert isinstance(result, ArtifactVersion)
            mock_grpc_stream.assert_called_once()

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('builtins.open', new_callable=mock_open, read_data=b'test content')
    def test_upload_success(self, mock_file, mock_getsize, mock_exists):
        """Test successful artifact version upload."""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024

        # Mock streaming response
        mock_response = Mock()
        mock_response.artifact_version_id = "new_version"
        mock_response.status.code = 10000  # SUCCESS

        with (
            patch('clarifai.client.base.BaseClient.__init__'),
            patch.object(ArtifactVersion, '_grpc_request_stream') as mock_grpc_stream,
        ):
            mock_grpc_stream.return_value = [mock_response]

            version = ArtifactVersion()
            result = version.upload(
                file_path="test_file.txt",
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app"
            )

            assert isinstance(result, ArtifactVersion)
            mock_grpc_stream.assert_called_once()

    @patch('os.path.exists')
    def test_upload_missing_file(self, mock_exists):
        """Test upload with missing file."""
        mock_exists.return_value = False

        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            with pytest.raises(UserError, match="File does not exist"):
                version.upload(
                    file_path="nonexistent_file.txt",
                    artifact_id="test_artifact",
                    user_id="test_user",
                    app_id="test_app"
                )

    @patch('requests.get')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_success(self, mock_file, mock_makedirs, mock_get):
        """Test successful artifact version download."""
        # Mock the info response
        mock_info_response = Mock()
        mock_info_response.status.code = 10000  # SUCCESS
        mock_info_response.artifact_version.upload.content_url = "https://example.com/file"
        mock_info_response.artifact_version.upload.content_name = "test_file.txt"
        mock_info_response.artifact_version.upload.content_length = 1024

        # Mock requests response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get.return_value = mock_response

        with (
            patch('clarifai.client.base.BaseClient.__init__'),
            patch.object(ArtifactVersion, '_grpc_request') as mock_grpc_request,
        ):
            mock_grpc_request.return_value = mock_info_response

            version = ArtifactVersion()
            result = version.download(
                output_path="./downloaded_file.txt",
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app"
            )

            assert result == "./downloaded_file.txt"
            mock_grpc_request.assert_called_once()
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == "GetArtifactVersion"

    def test_delete_success(self):
        """Test successful artifact version deletion."""
        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS

        with (
            patch('clarifai.client.base.BaseClient.__init__'),
            patch.object(ArtifactVersion, '_grpc_request') as mock_grpc_request,
        ):
            mock_grpc_request.return_value = mock_response

            version = ArtifactVersion()
            result = version.delete(
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app"
            )

            assert result is True
            mock_grpc_request.assert_called_once()
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == "DeleteArtifactVersion"

    def test_info_success(self):
        """Test successful artifact version info retrieval."""
        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS
        mock_response.artifact_version.id = "test_version"
        mock_response.artifact_version.description = "Test version"
        mock_response.artifact_version.visibility.name = "PRIVATE"
        mock_response.artifact_version.expires_at = None
        mock_response.artifact_version.created_at = None
        mock_response.artifact_version.modified_at = None
        mock_response.artifact_version.deleted_at = None
        mock_response.artifact_version.artifact = None
        mock_response.artifact_version.upload = None

        with (
            patch('clarifai.client.base.BaseClient.__init__'),
            patch.object(ArtifactVersion, '_grpc_request') as mock_grpc_request,
        ):
            mock_grpc_request.return_value = mock_response

            version = ArtifactVersion()
            info = version.info(
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app"
            )

            assert info["id"] == "test_version"
            assert info["description"] == "Test version"
            assert info["visibility"] == "PRIVATE"
            mock_grpc_request.assert_called_once()
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == "GetArtifactVersion"

    def test_list_success(self):
        """Test successful artifact version listing."""
        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS
        mock_version = Mock()
        mock_version.id = "test_version"
        mock_response.artifact_versions = [mock_version]

        with (
            patch('clarifai.client.base.BaseClient.__init__'),
            patch.object(ArtifactVersion, '_grpc_request') as mock_grpc_request,
        ):
            mock_grpc_request.return_value = mock_response

            versions = list(ArtifactVersion.list(
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app"
            ))

            assert len(versions) == 1
            assert versions[0].version_id == "test_version"
            mock_grpc_request.assert_called_once()
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == "ListArtifactVersions"


class TestArtifactVersionHelpers:
    """Test class for artifact version helper functions."""

    def test_format_bytes(self):
        """Test byte formatting utility."""
        assert format_bytes(0) == "0.0 B"
        assert format_bytes(512) == "512.0 B"
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1536) == "1.5 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert format_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TB"


class TestArtifactVersionValidation:
    """Test class for artifact version validation."""

    def test_create_missing_artifact_id(self):
        """Test creation with missing artifact_id."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            with pytest.raises(UserError, match="artifact_id is required"):
                version.create(
                    file_path="test_file.txt",
                    artifact_id="",
                    user_id="test_user",
                    app_id="test_app"
                )

    def test_create_missing_user_id(self):
        """Test creation with missing user_id."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            with pytest.raises(UserError, match="user_id is required"):
                version.create(
                    file_path="test_file.txt",
                    artifact_id="test_artifact",
                    user_id="",
                    app_id="test_app"
                )

    def test_create_missing_app_id(self):
        """Test creation with missing app_id."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            with pytest.raises(UserError, match="app_id is required"):
                version.create(
                    file_path="test_file.txt",
                    artifact_id="test_artifact",
                    user_id="test_user",
                    app_id=""
                )

    def test_delete_missing_version_id(self):
        """Test deletion with missing version_id."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            with pytest.raises(UserError, match="version_id is required"):
                version.delete(
                    artifact_id="test_artifact",
                    version_id="",
                    user_id="test_user",
                    app_id="test_app"
                )

    def test_info_missing_artifact_id(self):
        """Test info with missing artifact_id."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            with pytest.raises(UserError, match="artifact_id is required"):
                version.info(
                    artifact_id="",
                    version_id="test_version",
                    user_id="test_user",
                    app_id="test_app"
                )


if __name__ == "__main__":
    pytest.main([__file__])
