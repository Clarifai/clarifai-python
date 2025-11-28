"""Test file for artifact version functionality."""

from unittest.mock import Mock, mock_open, patch

import pytest
from google.protobuf import timestamp_pb2

from clarifai.client.artifact_version import ArtifactVersion, format_bytes
from clarifai.errors import UserError


class TestArtifactVersion:
    """Test class for ArtifactVersion client."""

    def test_init(self):
        """Test artifact version initialization."""
        with patch('clarifai.client.base.BaseClient.__init__'):
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
                user_id="test_user",
                app_id="test_app",
                base_url="https://api.clarifai.com"
            )
            assert version.artifact_id == "test_artifact"

    def test_repr(self):
        """Test artifact version string representation."""
        with patch('clarifai.client.base.BaseClient.__init__'):
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

    @patch('clarifai.client.artifact_version.handle_grpc_error')
    def test_create_success(self, mock_handle_grpc_error):
        """Test successful artifact version creation."""
        mock_response = Mock()
        mock_artifact_version = Mock()
        mock_artifact_version.id = "new_version"
        mock_response.artifact_versions = [mock_artifact_version]

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(ArtifactVersion, 'V2_STUB'):

            mock_handle_grpc_error.return_value = mock_response

            version = ArtifactVersion()
            result = version.create(
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app"
            )

            assert isinstance(result, ArtifactVersion)
            mock_handle_grpc_error.assert_called_once()

    def test_create_missing_params(self):
        """Test artifact version creation with missing parameters."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            with pytest.raises(UserError, match="artifact_id is required"):
                version.create()

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('builtins.open', new_callable=mock_open, read_data=b"test content")
    @patch('clarifai.client.artifact_version.handle_grpc_error')
    def test_upload_success(self, mock_handle_grpc_error, mock_file, mock_getsize, mock_exists):
        """Test successful file upload."""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024

        # Mock successful upload response
        mock_response = Mock()
        mock_artifact_version = Mock()
        mock_artifact_version.id = "uploaded_version"
        mock_response.artifact_versions = [mock_artifact_version]

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(ArtifactVersion, 'V2_STUB'), \
             patch.object(ArtifactVersion, '_streaming_upload_with_retry') as mock_upload:

            mock_handle_grpc_error.return_value = mock_response
            mock_upload.return_value = mock_response

            version = ArtifactVersion()
            result = version.upload(
                file_path="test_file.txt",
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app"
            )

            assert isinstance(result, ArtifactVersion)

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

    def test_upload_missing_params(self):
        """Test upload with missing required parameters."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            with pytest.raises(UserError, match="artifact_id is required"):
                version.upload(file_path="test.txt")

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    @patch('clarifai.client.artifact_version.handle_grpc_error')
    def test_download_success(self, mock_handle_grpc_error, mock_makedirs, mock_file):
        """Test successful file download."""
        # Mock download response with streaming data
        mock_response = Mock()
        mock_response.data.chunks = [b"chunk1", b"chunk2", b"chunk3"]

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(ArtifactVersion, 'V2_STUB'), \
             patch.object(ArtifactVersion, '_download_with_retry') as mock_download:

            mock_download.return_value = mock_response

            version = ArtifactVersion(
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app"
            )

            result = version.download(file_path="test_download.txt")
            assert result == "test_download.txt"

    def test_download_missing_params(self):
        """Test download with missing required parameters."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            with pytest.raises(UserError, match="artifact_id is required"):
                version.download(file_path="test.txt")

    @patch('clarifai.client.artifact_version.handle_grpc_error')
    def test_delete_success(self, mock_handle_grpc_error):
        """Test successful artifact version deletion."""
        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(ArtifactVersion, 'V2_STUB'):

            mock_handle_grpc_error.return_value = mock_response

            version = ArtifactVersion(
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app"
            )

            result = version.delete()
            assert result is True

    def test_delete_missing_params(self):
        """Test artifact version deletion with missing parameters."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            with pytest.raises(UserError, match="artifact_id is required"):
                version.delete()

    @patch('clarifai.client.artifact_version.handle_grpc_error')
    def test_info_success(self, mock_handle_grpc_error):
        """Test successful artifact version info retrieval."""
        mock_timestamp = timestamp_pb2.Timestamp()
        mock_timestamp.GetCurrentTime()

        mock_response = Mock()
        mock_artifact_version = Mock()
        mock_artifact_version.id = "test_version"
        mock_artifact_version.artifact_id = "test_artifact"
        mock_artifact_version.user_id = "test_user"
        mock_artifact_version.app_id = "test_app"
        mock_artifact_version.created_at = mock_timestamp
        mock_artifact_version.modified_at = mock_timestamp
        mock_response.artifact_versions = [mock_artifact_version]

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(ArtifactVersion, 'V2_STUB'):

            mock_handle_grpc_error.return_value = mock_response

            version = ArtifactVersion(
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app"
            )

            result = version.info()
            assert result is not None

    @patch('clarifai.client.artifact_version.handle_grpc_error')
    def test_list_success(self, mock_handle_grpc_error):
        """Test successful artifact version listing."""
        mock_response = Mock()
        mock_version1 = Mock()
        mock_version1.id = "version1"
        mock_version2 = Mock()
        mock_version2.id = "version2"
        mock_response.artifact_versions = [mock_version1, mock_version2]

        with patch('clarifai.client.base.BaseClient.__init__'), \
             patch.object(ArtifactVersion, 'V2_STUB'):

            mock_handle_grpc_error.return_value = mock_response

            version = ArtifactVersion()
            results = list(version.list(
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app"
            ))

            assert len(results) == 2

    def test_list_missing_params(self):
        """Test list with missing required parameters."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            with pytest.raises(UserError, match="artifact_id is required"):
                list(version.list())


class TestArtifactVersionHelpers:
    """Test helper functions for ArtifactVersion."""

    def test_format_bytes(self):
        """Test byte formatting function."""
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert format_bytes(512) == "512 B"
        assert format_bytes(0) == "0 B"

    def test_create_upload_config(self):
        """Test upload configuration creation."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            config = version._create_upload_config(
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app",
                version_id="test_version",
                description="Test description"
            )

            assert config.artifact_id == "test_artifact"
            assert config.user_id == "test_user"
            assert config.app_id == "test_app"

    @patch('os.path.getsize')
    @patch('builtins.open', new_callable=mock_open, read_data=b"test content")
    def test_artifact_version_upload_iterator(self, mock_file, mock_getsize):
        """Test upload iterator functionality."""
        mock_getsize.return_value = len(b"test content")

        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            # Create a mock upload config
            upload_config = Mock()
            upload_config.artifact_id = "test_artifact"

            iterator = version._artifact_version_upload_iterator(
                "test_file.txt", upload_config, chunk_size=4
            )

            chunks = list(iterator)
            assert len(chunks) >= 1  # At least the config chunk

    def test_get_client_params(self):
        """Test client parameter extraction."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion(
                artifact_id="test_artifact",
                version_id="test_version", 
                user_id="test_user",
                app_id="test_app"
            )

            params = version._get_client_params()
            expected = {
                'artifact_id': 'test_artifact',
                'version_id': 'test_version',
                'user_id': 'test_user',
                'app_id': 'test_app'
            }
            assert params == expected


class TestArtifactVersionValidation:
    """Test input validation for ArtifactVersion."""

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            # Test various missing parameter scenarios
            with pytest.raises(UserError, match="artifact_id is required"):
                version.create()

            with pytest.raises(UserError, match="artifact_id is required"):
                version.upload(file_path="test.txt")

            with pytest.raises(UserError, match="artifact_id is required"):
                version.delete()

    def test_invalid_file_paths(self):
        """Test validation with invalid file paths."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion()

            # Test empty file path
            with pytest.raises(UserError, match="file_path is required"):
                version.upload(
                    file_path="",
                    artifact_id="test_artifact",
                    user_id="test_user",
                    app_id="test_app"
                )


if __name__ == "__main__":
    pytest.main([__file__])
