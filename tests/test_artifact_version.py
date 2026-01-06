"""Test file for artifact version functionality."""

from unittest.mock import Mock, mock_open, patch

import pytest
import requests
from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf import timestamp_pb2

from clarifai.client.artifact_version import ArtifactVersion
from clarifai.errors import UserError
from clarifai.utils.misc import format_bytes
from tests.test_artifact_utils import create_mock_artifact_version


class TestArtifactVersion:
    """Test class for ArtifactVersion client."""

    def test_init(self):
        """Test artifact version initialization."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion(
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app",
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
                base_url="https://api.clarifai.com",
            )
            assert version.artifact_id == "test_artifact"

    def test_repr(self):
        """Test artifact version string representation."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            version = ArtifactVersion(
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app",
            )

            repr_str = repr(version)
            assert "test_artifact" in repr_str
            assert "test_version" in repr_str
            assert "test_user" in repr_str
            assert "test_app" in repr_str

    @patch('builtins.open', new_callable=mock_open, read_data=b"test file content")
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('clarifai.client.artifact.Artifact')
    def test_create_success(self, mock_artifact_class, mock_getsize, mock_exists, mock_file):
        """Test successful artifact version creation."""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024

        # Mock successful upload responses
        mock_response = Mock()
        mock_response.artifact_version_id = "new_version"
        mock_response.status.code = 10000  # SUCCESS

        # Mock the Artifact class and its instance
        mock_artifact = Mock()
        mock_artifact.get.side_effect = Exception("Not found")  # To trigger create
        mock_artifact.create.return_value = mock_artifact
        mock_artifact_class.return_value = mock_artifact

        with patch('clarifai.client.base.BaseClient.__init__', return_value=None):
            version = ArtifactVersion()

            # Mock the auth_helper and stub
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
                user_id="test_user", app_id="test_app"
            )
            mock_auth_helper.metadata = {}
            version.auth_helper = mock_auth_helper

            # Mock the streaming response as an iterator
            mock_stub = Mock()
            mock_stub.PostArtifactVersionsUpload.return_value = iter([mock_response])
            mock_auth_helper.get_stub.return_value = mock_stub
            version.STUB = mock_stub  # Add the STUB property

            result = version.upload(
                file_path="test_file.txt",
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app",
            )

            assert isinstance(result, ArtifactVersion)

    def test_create_missing_params(self):
        """Test artifact version creation with missing parameters."""
        version = create_mock_artifact_version()

        # Test missing artifact_id (first validation check)
        with pytest.raises(UserError, match="artifact_id is required"):
            version.upload(file_path="test.txt")

    @patch('builtins.open', new_callable=mock_open, read_data=b"test file content")
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_upload_success(self, mock_getsize, mock_exists, mock_file):
        """Test successful file upload."""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024

        # Mock successful upload response
        mock_response = Mock()
        mock_response.artifact_version_id = "uploaded_version"
        mock_response.status.code = 10000  # SUCCESS

        with (
            patch('clarifai.client.base.BaseClient.__init__', return_value=None),
            patch('clarifai.client.artifact.Artifact') as mock_artifact_class,
        ):
            # Mock the artifact for auto-creation in upload
            mock_artifact = Mock()
            mock_artifact.get.side_effect = Exception(
                "Not found"
            )  # Simulate artifact doesn't exist
            mock_artifact.create.return_value = Mock()
            mock_artifact_class.return_value = mock_artifact

            version = ArtifactVersion()

            # Mock the auth_helper and stub
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
                user_id="test_user", app_id="test_app"
            )
            mock_auth_helper.metadata = {}
            version.auth_helper = mock_auth_helper

            # Mock the streaming response as an iterator
            mock_stub = Mock()
            mock_stub.PostArtifactVersionsUpload.return_value = iter([mock_response])
            version.STUB = mock_stub

            result = version.upload(
                file_path="test_file.txt",
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app",
            )

            assert isinstance(result, ArtifactVersion)

    @patch('os.path.exists')
    def test_upload_missing_file(self, mock_exists):
        """Test upload with missing file."""
        mock_exists.return_value = False

        version = create_mock_artifact_version()

        with pytest.raises(UserError, match="File does not exist"):
            version.upload(
                file_path="nonexistent_file.txt",
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app",
            )

    def test_upload_missing_params(self):
        """Test upload with missing required parameters."""
        version = create_mock_artifact_version()

        # Test missing artifact_id
        with pytest.raises(UserError, match="artifact_id is required"):
            version.upload(file_path="test.txt", user_id="test_user", app_id="test_app")

        # Test missing user_id
        with pytest.raises(UserError, match="user_id is required"):
            version.upload(file_path="test.txt", artifact_id="test_artifact", app_id="test_app")

        # Test missing app_id
        with pytest.raises(UserError, match="app_id is required"):
            version.upload(file_path="test.txt", artifact_id="test_artifact", user_id="test_user")

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    @patch('requests.get')
    def test_download_success(self, mock_requests_get, mock_makedirs, mock_file):
        """Test successful file download."""
        # Mock the info response first
        mock_info_response = Mock()
        mock_info_response.status.code = 10000  # SUCCESS

        # Create proper mock artifact version with upload info
        mock_artifact_version = Mock()
        mock_artifact_version.id = "test_version"
        mock_upload = Mock()
        mock_upload.content_url = "https://example.com/file.txt"
        mock_upload.content_name = "test_file.txt"
        mock_upload.content_length = 1024
        mock_artifact_version.upload = mock_upload
        mock_info_response.artifact_version = mock_artifact_version

        # Mock the HTTP request response
        mock_http_response = Mock()
        mock_http_response.status_code = 200
        mock_http_response.headers = {'content-length': '1024'}
        mock_http_response.iter_content.return_value = [b'test content chunk']
        mock_http_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_http_response

        with (
            patch('clarifai.client.base.BaseClient.__init__', return_value=None),
            patch.object(ArtifactVersion, '_grpc_request') as mock_grpc_request,
            patch('os.path.exists', return_value=False),  # File doesn't exist
            patch('os.path.getsize', return_value=1024),
        ):
            mock_grpc_request.return_value = mock_info_response

            version = ArtifactVersion(
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app",
            )
            # Mock the auth_helper and STUB attributes
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
                user_id="test_user", app_id="test_app"
            )
            version.auth_helper = mock_auth_helper
            mock_stub = Mock()
            mock_stub.GetArtifactVersion = Mock()
            version.STUB = mock_stub

            result = version.download(output_path="test_download.txt", force=True)
            assert result == "test_download.txt"

            # Verify the actual HTTP request was made
            mock_requests_get.assert_called_once_with(
                "https://example.com/file.txt", stream=True, headers={}
            )
            mock_grpc_request.assert_called_once()

    @patch('requests.get')
    def test_download_http_403_failure(self, mock_requests_get):
        """Test download failure with 403 Forbidden error."""
        # Mock the info response first
        mock_info_response = Mock()
        mock_info_response.status.code = 10000  # SUCCESS

        # Create proper mock artifact version with upload info
        mock_artifact_version = Mock()
        mock_artifact_version.id = "test_version"
        mock_upload = Mock()
        mock_upload.content_url = "https://s3.amazonaws.com/clarifai-mdata-prod-virginia-models-build-context/uploads/prod/655fd751d4a14aeab312eb42a3366655/776b442bb41244ac8ce848b801add297/artifact_test-version-1"
        mock_upload.content_name = "test_file.txt"
        mock_upload.content_length = 1024
        mock_artifact_version.upload = mock_upload
        mock_info_response.artifact_version = mock_artifact_version

        # Mock the HTTP request to raise 403 error
        mock_http_response = Mock()
        mock_http_response.raise_for_status.side_effect = requests.HTTPError(
            "403 Client Error: Forbidden for url"
        )
        mock_requests_get.return_value = mock_http_response

        with (
            patch('clarifai.client.base.BaseClient.__init__', return_value=None),
            patch.object(ArtifactVersion, '_grpc_request') as mock_grpc_request,
            patch('os.path.exists', return_value=False),  # File doesn't exist
        ):
            mock_grpc_request.return_value = mock_info_response

            version = ArtifactVersion(
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app",
            )
            # Mock the auth_helper and STUB attributes
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
                user_id="test_user", app_id="test_app"
            )
            version.auth_helper = mock_auth_helper
            mock_stub = Mock()
            mock_stub.GetArtifactVersion = Mock()
            version.STUB = mock_stub

            # This should now fail as expected
            with pytest.raises(
                UserError, match="Download failed due to network error.*403.*Forbidden"
            ):
                version.download(output_path="test_download.txt", force=True)

    def test_download_missing_params(self):
        """Test download with missing required parameters."""
        version = create_mock_artifact_version()

        with pytest.raises(UserError, match="artifact_id is required"):
            version.download(output_path="test.txt")

    def test_delete_success(self):
        """Test successful artifact version deletion."""
        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS

        with (
            patch('clarifai.client.base.BaseClient.__init__', return_value=None),
            patch.object(ArtifactVersion, '_grpc_request') as mock_grpc_request,
        ):
            mock_grpc_request.return_value = mock_response

            version = ArtifactVersion(
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app",
            )
            # Mock the auth_helper attribute
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
                user_id="test_user", app_id="test_app"
            )
            version.auth_helper = mock_auth_helper
            mock_stub = Mock()
            mock_stub.DeleteArtifactVersion = Mock()
            version.STUB = mock_stub

            result = version.delete()
            assert result is True
            mock_grpc_request.assert_called_once()
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == mock_stub.DeleteArtifactVersion

    def test_delete_missing_params(self):
        """Test artifact version deletion with missing parameters."""
        version = create_mock_artifact_version()

        with pytest.raises(UserError, match="artifact_id is required"):
            version.delete()

    def test_get_success(self):
        """Test successful artifact version get retrieval."""
        mock_timestamp = timestamp_pb2.Timestamp()
        mock_timestamp.GetCurrentTime()

        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS
        mock_response.artifact_version.id = "test_version"
        mock_response.artifact_version.artifact_id = "test_artifact"
        mock_response.artifact_version.user_id = "test_user"
        mock_response.artifact_version.app_id = "test_app"
        mock_response.artifact_version.created_at = mock_timestamp
        mock_response.artifact_version.modified_at = mock_timestamp

        with (
            patch('clarifai.client.base.BaseClient.__init__', return_value=None),
            patch.object(ArtifactVersion, '_grpc_request') as mock_grpc_request,
        ):
            mock_grpc_request.return_value = mock_response

            version = ArtifactVersion(
                artifact_id="test_artifact",
                version_id="test_version",
                user_id="test_user",
                app_id="test_app",
            )
            # Mock the auth_helper and STUB attributes
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
                user_id="test_user", app_id="test_app"
            )
            version.auth_helper = mock_auth_helper
            mock_stub = Mock()
            mock_stub.GetArtifactVersion = Mock()
            version.STUB = mock_stub

            result = version.get()
            assert result is not None
            mock_grpc_request.assert_called_once()
            call_args = mock_grpc_request.call_args
            assert call_args[0][0] == mock_stub.GetArtifactVersion

    def test_list_success(self):
        """Test successful artifact version listing."""
        mock_response = Mock()
        mock_response.status.code = 10000  # SUCCESS
        mock_version1 = Mock()
        mock_version1.id = "version1"
        mock_version2 = Mock()
        mock_version2.id = "version2"
        mock_response.artifact_versions = [mock_version1, mock_version2]

        # Since list() is instance method now, we need to patch the instance
        with (
            patch('clarifai.client.base.BaseClient.__init__', return_value=None),
            patch.object(ArtifactVersion, '_grpc_request') as mock_grpc_request,
        ):
            mock_grpc_request.return_value = mock_response

            version = ArtifactVersion(
                artifact_id="test_artifact", user_id="test_user", app_id="test_app"
            )
            # Mock the auth_helper and STUB attributes
            mock_auth_helper = Mock()
            mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
                user_id="test_user", app_id="test_app"
            )
            version.auth_helper = mock_auth_helper
            mock_stub = Mock()
            mock_stub.ListArtifactVersions = Mock()
            version.STUB = mock_stub

            results = list(version.list())

            assert len(results) == 2
            assert results[0].id == "version1"
            assert results[1].id == "version2"
            mock_grpc_request.assert_called_once()

    def test_list_missing_params(self):
        """Test list with missing required parameters."""
        with pytest.raises(UserError, match="artifact_id is required"):
            list(ArtifactVersion().list())

    def test_upload_missing_artifact_id(self):
        """Test upload with missing artifact_id (now required)."""
        version = create_mock_artifact_version()

        # Test missing artifact_id
        with pytest.raises(UserError, match="artifact_id is required"):
            version.upload(
                file_path="test.txt",
                artifact_id="",  # Empty string should trigger error
                user_id="test_user",
                app_id="test_app",
            )


class TestArtifactVersionHelpers:
    """Test helper functions for ArtifactVersion."""

    def test_format_bytes(self):
        """Test byte formatting function."""
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert format_bytes(512) == "512.0 B"
        assert format_bytes(0) == "0 B"

    def test_create_upload_config(self):
        """Test upload configuration creation."""
        version = create_mock_artifact_version()

        config = version._create_upload_config(
            artifact_id="test_artifact",
            description="Test description",
            visibility="private",
            expires_at=None,
            version_id="test_version",
            user_id="test_user",
            app_id="test_app",
            file_size=1024,
        )

        assert config.upload_config.artifact_id == "test_artifact"
        assert config.upload_config.user_app_id.user_id == "test_user"
        assert config.upload_config.user_app_id.app_id == "test_app"
        assert config.upload_config.artifact_version.id == "test_version"
        assert config.upload_config.artifact_version.description == "Test description"

    def test_create_upload_config_visibility_options(self):
        """Test upload configuration with different visibility options."""
        from clarifai_grpc.grpc.api import resources_pb2

        version = create_mock_artifact_version()

        # Test private visibility
        config = version._create_upload_config(
            artifact_id="test_artifact",
            description="Test description",
            visibility="private",
            expires_at=None,
            version_id="test_version",
            user_id="test_user",
            app_id="test_app",
            file_size=1024,
        )
        assert (
            config.upload_config.artifact_version.visibility.gettable
            == resources_pb2.Visibility.Gettable.PRIVATE
        )

        # Test public visibility
        config = version._create_upload_config(
            artifact_id="test_artifact",
            description="Test description",
            visibility="public",
            expires_at=None,
            version_id="test_version",
            user_id="test_user",
            app_id="test_app",
            file_size=1024,
        )
        assert (
            config.upload_config.artifact_version.visibility.gettable
            == resources_pb2.Visibility.Gettable.PUBLIC
        )

        # Test org visibility
        config = version._create_upload_config(
            artifact_id="test_artifact",
            description="Test description",
            visibility="org",
            expires_at=None,
            version_id="test_version",
            user_id="test_user",
            app_id="test_app",
            file_size=1024,
        )
        assert (
            config.upload_config.artifact_version.visibility.gettable
            == resources_pb2.Visibility.Gettable.ORG
        )

        # Test invalid visibility raises error
        with pytest.raises(UserError, match="Invalid visibility value: 'invalid'"):
            version._create_upload_config(
                artifact_id="test_artifact",
                description="Test description",
                visibility="invalid",
                expires_at=None,
                version_id="test_version",
                user_id="test_user",
                app_id="test_app",
                file_size=1024,
            )

    @patch('os.path.getsize')
    @patch('builtins.open', new_callable=mock_open, read_data=b"test content")
    def test_artifact_version_upload_iterator(self, mock_file, mock_getsize):
        """Test upload iterator functionality."""
        mock_getsize.return_value = len(b"test content")

        version = create_mock_artifact_version()

        iterator = version._artifact_version_upload_iterator(
            file_path="test_file.txt",
            artifact_id="test_artifact",
            description="Test description",
            visibility="private",
            expires_at=None,
            version_id="test_version",
            user_id="test_user",
            app_id="test_app",
        )

        chunks = list(iterator)
        assert len(chunks) >= 1  # At least the config chunk


class TestArtifactVersionValidation:
    """Test input validation for ArtifactVersion."""

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        version = create_mock_artifact_version()

        # Test various missing parameter scenarios - artifact_id is now checked first in upload()
        with pytest.raises(UserError, match="artifact_id is required"):
            version.upload(file_path="test.txt")

        # artifact_id is also required for delete operations
        with pytest.raises(UserError, match="artifact_id is required"):
            version.delete()

    def test_invalid_file_paths(self):
        """Test validation with invalid file paths."""
        version = create_mock_artifact_version()

        # Test empty file path
        with pytest.raises(UserError, match="file_path is required"):
            version.upload(
                file_path="",
                artifact_id="test_artifact",
                user_id="test_user",
                app_id="test_app",
            )
