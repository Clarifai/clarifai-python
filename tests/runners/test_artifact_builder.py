"""Test file for artifact builder functionality."""

from unittest.mock import Mock, patch

import pytest

from clarifai.cli.artifact import is_local_path
from clarifai.errors import UserError
from clarifai.runners.artifacts.artifact_builder import (
    ArtifactBuilder,
    download_artifact,
    parse_artifact_path,
    upload_artifact,
)


class TestArtifactPathParsing:
    """Test class for artifact path parsing utilities."""

    def test_parse_valid_artifact_path(self):
        """Test parsing valid artifact paths."""
        path = "users/u123/apps/a456/artifacts/my_artifact"
        parsed = parse_artifact_path(path)

        assert parsed['user_id'] == 'u123'
        assert parsed['app_id'] == 'a456'
        assert parsed['artifact_id'] == 'my_artifact'
        assert parsed['version_id'] is None

    def test_parse_valid_version_path(self):
        """Test parsing valid artifact version paths."""
        path = "users/u123/apps/a456/artifacts/my_artifact/versions/v789"
        parsed = parse_artifact_path(path)

        assert parsed['user_id'] == 'u123'
        assert parsed['app_id'] == 'a456'
        assert parsed['artifact_id'] == 'my_artifact'
        assert parsed['version_id'] == 'v789'

    def test_parse_path_with_trailing_slash(self):
        """Test parsing paths with trailing slash."""
        path = "users/u123/apps/a456/artifacts/my_artifact/"
        parsed = parse_artifact_path(path)

        assert parsed['user_id'] == 'u123'
        assert parsed['app_id'] == 'a456'
        assert parsed['artifact_id'] == 'my_artifact'
        assert parsed['version_id'] is None

    def test_parse_invalid_path(self):
        """Test parsing invalid paths."""
        invalid_paths = [
            "invalid/path",
            "users/u123/invalid/path",
            "users/u123/apps/a456/invalid",
            "",
            "users/",
            "users/u123/apps/a456/artifacts/",  # Missing artifact ID
        ]

        for invalid_path in invalid_paths:
            with pytest.raises(UserError, match="Invalid artifact path format"):
                parse_artifact_path(invalid_path)

    def test_parse_path_with_special_characters(self):
        """Test parsing paths with special characters."""
        path = "users/user-123/apps/app_456/artifacts/my-artifact_v2"
        parsed = parse_artifact_path(path)

        assert parsed['user_id'] == 'user-123'
        assert parsed['app_id'] == 'app_456'
        assert parsed['artifact_id'] == 'my-artifact_v2'

    def test_is_local_path(self):
        """Test local path detection."""
        # Positive cases
        assert is_local_path("./local/file.txt") is True
        assert is_local_path("/home/user/file.txt") is True
        assert is_local_path("file.txt") is True
        assert is_local_path("../parent/file.txt") is True
        assert is_local_path("~/home/file.txt") is True

        # Negative cases
        assert is_local_path("users/u123/apps/a456/artifacts/my_artifact") is False
        assert is_local_path("http://example.com/file.txt") is False
        assert is_local_path("ftp://example.com/file.txt") is False

    def test_parse_timestamp_to_version_id(self):
        """Test timestamp parsing for version IDs."""
        # This function doesn't exist yet, so we'll test the concept
        # In the future, this could be implemented for RFC3339 timestamp parsing
        pass


class TestArtifactBuilder:
    """Test class for ArtifactBuilder."""

    def setup_method(self):
        """Setup for each test method."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            self.builder = ArtifactBuilder()
            # Mock the auth_helper and base attributes
            mock_auth_helper = Mock()
            mock_auth_helper.pat = "mock_pat"
            self.builder.auth_helper = mock_auth_helper
            self.builder.base = "https://api.clarifai.com"

    def test_init(self):
        """Test ArtifactBuilder initialization."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            builder = ArtifactBuilder()
            assert builder is not None

    @patch('os.path.exists')
    @patch('clarifai.client.artifact.Artifact.create')
    @patch('clarifai.client.artifact_version.ArtifactVersion.upload')
    def test_upload_from_path_success(self, mock_upload, mock_create, mock_exists):
        """Test successful upload from path."""
        mock_exists.return_value = True
        mock_create.return_value = Mock()
        mock_upload.return_value = Mock(id="new_version")

        result = self.builder.upload_from_path(
            source_path="./test_file.txt",
            destination_path="users/u123/apps/a456/artifacts/my_artifact",
        )

        assert result.id == "new_version"
        mock_upload.assert_called_once()

    @patch('os.path.exists')
    def test_upload_from_path_missing_file(self, mock_exists):
        """Test upload from path with missing source file."""
        mock_exists.return_value = False

        with pytest.raises(UserError, match="Source file does not exist"):
            self.builder.upload_from_path(
                source_path="./nonexistent_file.txt",
                destination_path="users/u123/apps/a456/artifacts/my_artifact",
            )

    def test_upload_from_path_invalid_destination(self):
        """Test upload from path with invalid destination path."""
        with pytest.raises(UserError, match="destination_path must be an artifact path"):
            self.builder.upload_from_path(
                source_path="./test_file.txt", destination_path="invalid/path"
            )

    @patch('os.path.exists')
    @patch('clarifai.client.artifact.Artifact.create')
    @patch('clarifai.client.artifact_version.ArtifactVersion.upload')
    def test_upload_from_path_with_description(self, mock_upload, mock_create, mock_exists):
        """Test upload from path with custom description."""
        mock_exists.return_value = True
        mock_create.return_value = Mock()
        mock_upload.return_value = Mock(id="new_version")

        self.builder.upload_from_path(
            source_path="./test_file.txt",
            destination_path="users/u123/apps/a456/artifacts/my_artifact",
            description="Custom version description",
        )

        # Verify upload was called with description
        call_args = mock_upload.call_args
        assert "Custom version description" in str(call_args)

    @patch('clarifai.client.artifact_version.ArtifactVersion.download')
    def test_download_from_path_success(self, mock_download):
        """Test successful download from path."""
        mock_download.return_value = "./downloaded_file.txt"

        result = self.builder.download_from_path(
            source_path="users/u123/apps/a456/artifacts/my_artifact",
            destination_path="./downloaded_file.txt",
        )

        assert result == "./downloaded_file.txt"
        mock_download.assert_called_once()

    def test_download_from_path_invalid_source(self):
        """Test download from path with invalid source path."""
        with pytest.raises(UserError, match="source_path must be an artifact path"):
            self.builder.download_from_path(
                source_path="invalid/path", destination_path="./downloaded_file.txt"
            )

    @patch('clarifai.client.artifact_version.ArtifactVersion.download')
    def test_download_from_path_with_version(self, mock_download):
        """Test download from path with specific version."""
        mock_download.return_value = "./downloaded_file.txt"

        result = self.builder.download_from_path(
            source_path="users/u123/apps/a456/artifacts/my_artifact/versions/v789",
            destination_path="./downloaded_file.txt",
        )

        assert result == "./downloaded_file.txt"
        # Verify the version was passed correctly
        call_args = mock_download.call_args
        assert "v789" in str(call_args)

    @patch('clarifai.client.artifact.Artifact.list')
    def test_list_artifacts(self, mock_list):
        """Test listing artifacts."""
        # Mock artifact objects
        mock_artifact1 = Mock()
        mock_artifact1.id = "artifact1"
        mock_artifact1.description = "Test artifact 1"

        mock_artifact2 = Mock()
        mock_artifact2.id = "artifact2"
        mock_artifact2.description = "Test artifact 2"

        mock_list.return_value = [mock_artifact1, mock_artifact2]

        result = self.builder.list_artifacts(user_id="u123", app_id="a456")

        artifacts = list(result)
        assert len(artifacts) == 2
        assert artifacts[0].id == "artifact1"
        assert artifacts[1].id == "artifact2"

    @patch('clarifai.client.artifact_version.ArtifactVersion.list')
    def test_list_artifact_versions(self, mock_list):
        """Test listing artifact versions."""
        # Mock version objects
        mock_version1 = Mock()
        mock_version1.id = "version1"
        mock_version1.artifact_id = "test_artifact"

        mock_version2 = Mock()
        mock_version2.id = "version2"
        mock_version2.artifact_id = "test_artifact"

        mock_list.return_value = [mock_version1, mock_version2]

        result = self.builder.list_artifact_versions(
            artifact_id="test_artifact", user_id="u123", app_id="a456"
        )

        versions = list(result)
        assert len(versions) == 2
        assert versions[0].id == "version1"
        assert versions[1].id == "version2"

    @patch('clarifai.client.artifact.Artifact.info')
    def test_get_artifact_info(self, mock_info):
        """Test getting artifact info."""
        mock_info.return_value = {'id': "test_artifact", 'description': "Test description"}

        result = self.builder.get_artifact_info(
            artifact_id="test_artifact", user_id="u123", app_id="a456"
        )

        assert result['id'] == "test_artifact"
        assert result['description'] == "Test description"

    @patch('clarifai.client.artifact_version.ArtifactVersion.info')
    def test_get_artifact_version_info(self, mock_info):
        """Test getting artifact version info."""
        mock_info.return_value = {'id': "test_version", 'artifact_id': "test_artifact"}

        result = self.builder.get_artifact_version_info(
            artifact_id="test_artifact", version_id="test_version", user_id="u123", app_id="a456"
        )

        assert result['id'] == "test_version"
        assert result['artifact_id'] == "test_artifact"

    @patch('clarifai.client.artifact.Artifact.delete')
    def test_delete_artifact(self, mock_delete):
        """Test deleting artifact."""
        mock_delete.return_value = True

        result = self.builder.delete_artifact(
            artifact_id="test_artifact", user_id="u123", app_id="a456"
        )

        assert result is True
        mock_delete.assert_called_once()

    @patch('clarifai.client.artifact_version.ArtifactVersion.delete')
    def test_delete_artifact_version(self, mock_delete):
        """Test deleting artifact version."""
        mock_delete.return_value = True

        result = self.builder.delete_artifact_version(
            artifact_id="test_artifact", version_id="test_version", user_id="u123", app_id="a456"
        )

        assert result is True
        mock_delete.assert_called_once()


class TestArtifactBuilderErrorHandling:
    """Test error handling in ArtifactBuilder."""

    def setup_method(self):
        """Setup for each test method."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            self.builder = ArtifactBuilder()

    def test_missing_required_parameters(self):
        """Test error handling for missing required parameters."""
        # Test upload without required parameters
        with pytest.raises(UserError, match="source_path is required"):
            self.builder.upload_from_path(
                source_path="", destination_path="users/u123/apps/a456/artifacts/my_artifact"
            )

        with pytest.raises(UserError, match="destination_path is required"):
            self.builder.upload_from_path(source_path="./test_file.txt", destination_path="")

    def test_invalid_paths_error_handling(self):
        """Test error handling for invalid paths."""
        # Test upload with both local paths
        with pytest.raises(UserError, match="destination_path must be an artifact path"):
            self.builder.upload_from_path(
                source_path="./file1.txt", destination_path="./file2.txt"
            )

        # Test download with both remote paths
        with pytest.raises(UserError, match="destination_path must be a local path"):
            self.builder.download_from_path(
                source_path="users/u123/apps/a456/artifacts/art1",
                destination_path="users/u123/apps/a456/artifacts/art2",
            )


class TestConvenienceFunctions:
    """Test convenience functions for artifact operations."""

    @patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder.upload_from_path')
    @patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder.__init__', return_value=None)
    def test_upload_artifact_convenience(self, mock_init, mock_upload):
        """Test upload_artifact convenience function."""
        mock_upload.return_value = Mock(id="new_version")

        result = upload_artifact(
            source_path="./test_file.txt",
            destination_path="users/u123/apps/a456/artifacts/my_artifact",
        )

        assert result.id == "new_version"
        mock_upload.assert_called_once()

    @patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder.download_from_path')
    @patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder.__init__', return_value=None)
    def test_download_artifact_convenience(self, mock_init, mock_download):
        """Test download_artifact convenience function."""
        mock_download.return_value = "./downloaded_file.txt"

        result = download_artifact(
            source_path="users/u123/apps/a456/artifacts/my_artifact",
            destination_path="./downloaded_file.txt",
        )

        assert result == "./downloaded_file.txt"
        mock_download.assert_called_once()


class TestArtifactBuilderIntegration:
    """Integration tests for ArtifactBuilder workflows."""

    def setup_method(self):
        """Setup for each test method."""
        with patch('clarifai.client.base.BaseClient.__init__'):
            self.builder = ArtifactBuilder()

    @patch('os.path.exists')
    @patch('clarifai.client.artifact.Artifact.create')
    @patch('clarifai.client.artifact_version.ArtifactVersion.upload')
    @patch('clarifai.client.artifact_version.ArtifactVersion.list')
    @patch('clarifai.client.artifact_version.ArtifactVersion.download')
    def test_complete_workflow_simulation(
        self, mock_download, mock_list, mock_upload, mock_create, mock_exists
    ):
        """Test complete artifact workflow simulation."""
        mock_exists.return_value = True
        mock_create.return_value = Mock()

        # 1. Upload artifact
        mock_upload.return_value = Mock(id="v1")
        result = self.builder.upload_from_path(
            source_path="./test_file.txt",
            destination_path="users/u123/apps/a456/artifacts/my_artifact",
        )
        assert result.id == "v1"

        # 2. List versions
        mock_version = Mock()
        mock_version.id = "v1"
        mock_list.return_value = [mock_version]

        versions = list(
            self.builder.list_artifact_versions(
                artifact_id="my_artifact", user_id="u123", app_id="a456"
            )
        )
        assert len(versions) == 1
        assert versions[0]['id'] == "v1"

        # 3. Download artifact
        mock_download.return_value = "./downloaded_file.txt"
        result = self.builder.download_from_path(
            source_path="users/u123/apps/a456/artifacts/my_artifact/versions/v1",
            destination_path="./downloaded_file.txt",
        )
        assert result == "./downloaded_file.txt"


if __name__ == "__main__":
    pytest.main([__file__])
