"""Test file for artifact version CLI functionality."""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from clarifai.cli.artifact import artifact
from clarifai.errors import UserError
from tests.test_artifact_utils import setup_context_mock


class TestArtifactVersionCLI:
    """Test class for artifact version CLI commands."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact_version.ArtifactVersion')
    def test_list_versions_command_success(self, mock_artifact_version_class, mock_validate):
        """Test successful list versions command."""
        mock_obj = setup_context_mock(mock_validate)

        # Mock the artifact version instance and its list method
        mock_version_instance = Mock()
        mock_version_list = []
        # Create mock version objects
        for i, version_id in enumerate(['version1', 'version2'], 1):
            mock_version = Mock()
            mock_version.id = version_id
            mock_version.description = f"Test version {i}"
            mock_version.visibility.gettable = 10  # PRIVATE enum value
            mock_version.created_at.ToDatetime.return_value = f"2023-01-0{i} 00:00:00"
            # Mock expires_at as None (never expires)
            mock_version.expires_at = None
            mock_version_list.append(mock_version)

        mock_version_instance.list.return_value = mock_version_list
        mock_artifact_version_class.return_value = mock_version_instance

        result = self.runner.invoke(
            artifact,
            ['list', 'users/test_user/apps/test_app/artifacts/test_artifact', '--versions'],
            obj=mock_obj,
        )

        assert result.exit_code == 0
        mock_artifact_version_class.assert_called_once()
        mock_version_instance.list.assert_called_once()

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact_version.ArtifactVersion')
    def test_get_version_command_success(self, mock_artifact_version_class, mock_validate):
        """Test successful get version command."""
        mock_obj = setup_context_mock(mock_validate)

        # Mock the artifact version instance and its get method
        mock_version_instance = Mock()
        mock_version_info = Mock()
        mock_version_info.id = 'test_version'
        mock_version_info.artifact_id = 'test_artifact'
        mock_version_info.description = 'Test version description'
        mock_version_info.visibility.gettable = 10  # PRIVATE enum value
        mock_version_info.expires_at = None  # Never expires
        mock_version_info.created_at.ToDatetime.return_value = '2023-01-01 00:00:00'
        mock_version_info.modified_at.ToDatetime.return_value = '2023-01-01 00:00:00'
        mock_version_info.upload.id = 'upload_123'
        mock_version_info.upload.content_name = 'test_file.txt'
        mock_version_info.upload.content_length = 1024
        mock_version_info.upload.status.description = 'Upload completed'

        mock_version_instance.get.return_value = mock_version_info
        mock_artifact_version_class.return_value = mock_version_instance

        result = self.runner.invoke(
            artifact,
            ['get', 'users/test_user/apps/test_app/artifacts/test_artifact/versions/test_version'],
            obj=mock_obj,
        )

        assert result.exit_code == 0
        mock_artifact_version_class.assert_called_once()
        mock_version_instance.get.assert_called_once()

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact_version.ArtifactVersion')
    def test_delete_version_command_success(self, mock_artifact_version_class, mock_validate):
        """Test successful delete version command."""
        mock_obj = setup_context_mock(mock_validate)

        # Mock the artifact version instance and its delete method
        mock_version_instance = Mock()
        mock_version_instance.delete.return_value = True
        mock_artifact_version_class.return_value = mock_version_instance

        result = self.runner.invoke(
            artifact,
            [
                'delete',
                'users/test_user/apps/test_app/artifacts/test_artifact/versions/test_version',
            ],
            input='y\n',
            obj=mock_obj,
        )

        assert result.exit_code == 0
        mock_artifact_version_class.assert_called_once()
        mock_version_instance.delete.assert_called_once()

    @patch('clarifai.cli.artifact.validate_context')
    def test_delete_version_command_cancel(self, mock_validate):
        """Test delete version command with user cancellation."""
        mock_obj = setup_context_mock(mock_validate)

        result = self.runner.invoke(
            artifact,
            [
                'delete',
                'users/test_user/apps/test_app/artifacts/test_artifact/versions/test_version',
            ],
            input='n\n',
            obj=mock_obj,
        )

        assert result.exit_code == 0
        assert "Operation cancelled" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    @patch('os.path.exists')
    def test_cp_upload_to_version_success(self, mock_exists, mock_validate):
        """Test successful upload to specific version via cp command."""
        mock_obj = setup_context_mock(mock_validate)
        mock_exists.return_value = True

        with patch('clarifai.client.artifact_version.ArtifactVersion') as mock_artifact_version:
            mock_instance = Mock()
            mock_artifact_version.return_value = mock_instance
            mock_instance.upload.return_value = Mock(id="new_version")

            result = self.runner.invoke(
                artifact,
                [
                    'cp',
                    './test_file.txt',
                    'users/test_user/apps/test_app/artifacts/test_artifact',
                    '--description',
                    'New version upload',
                ],
                obj=mock_obj,
            )

            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_cp_download_specific_version_success(self, mock_validate):
        """Test successful download of specific version via cp command."""
        mock_obj = setup_context_mock(mock_validate)

        with patch('clarifai.client.artifact_version.ArtifactVersion') as mock_artifact_version:
            mock_instance = Mock()
            mock_artifact_version.return_value = mock_instance
            mock_instance.download.return_value = "./downloaded_version.txt"

            result = self.runner.invoke(
                artifact,
                [
                    'cp',
                    'users/test_user/apps/test_app/artifacts/test_artifact/versions/test_version',
                    './downloaded_version.txt',
                ],
                obj=mock_obj,
            )

            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_invalid_version_path_handling(self, mock_validate):
        """Test handling of invalid version paths."""
        mock_obj = setup_context_mock(mock_validate)

        # Test invalid version path format
        result = self.runner.invoke(
            artifact,
            ['get', 'users/test_user/apps/test_app/artifacts/test_artifact/invalid_versions_path'],
            obj=mock_obj,
        )

        assert result.exit_code != 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_version_error_handling(self, mock_validate):
        """Test version-specific error handling."""
        mock_obj = setup_context_mock(mock_validate)

        with patch('clarifai.client.artifact_version.ArtifactVersion') as mock_artifact_version:
            mock_instance = Mock()
            mock_artifact_version.return_value = mock_instance

            # Simulate version not found error
            mock_instance.get.side_effect = UserError("Version not found")

            result = self.runner.invoke(
                artifact,
                [
                    'get',
                    'users/test_user/apps/test_app/artifacts/test_artifact/versions/nonexistent',
                ],
                obj=mock_obj,
            )

            assert result.exit_code != 0
            assert "Version not found" in result.output


class TestArtifactVersionCLIEdgeCases:
    """Test edge cases for artifact version CLI commands."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    @patch('clarifai.cli.artifact.validate_context')
    def test_list_versions_empty_result(self, mock_validate):
        """Test list versions with empty result."""
        mock_obj = setup_context_mock(mock_validate)

        with patch('clarifai.client.artifact_version.ArtifactVersion') as mock_artifact_version:
            mock_instance = Mock()
            mock_artifact_version.return_value = mock_instance
            mock_instance.list.return_value = []

            result = self.runner.invoke(
                artifact,
                ['list', 'users/test_user/apps/test_app/artifacts/test_artifact', '--versions'],
                obj=mock_obj,
            )

            assert result.exit_code == 0
            assert "No artifact versions found" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    def test_get_version_missing_version_id(self, mock_validate):
        """Test get command with missing version ID in path."""
        mock_obj = setup_context_mock(mock_validate)

        # Path that ends with '/versions/' but no version ID
        result = self.runner.invoke(
            artifact,
            ['get', 'users/test_user/apps/test_app/artifacts/test_artifact/versions/'],
            obj=mock_obj,
        )

        assert result.exit_code != 0

    @patch('clarifai.cli.artifact.validate_context')
    @patch('os.path.exists')
    def test_upload_large_file_simulation(self, mock_exists, mock_validate):
        """Test upload simulation for large files."""
        mock_obj = setup_context_mock(mock_validate)
        mock_exists.return_value = True

        with (
            patch('clarifai.client.artifact_version.ArtifactVersion') as mock_artifact_version,
            patch('os.path.getsize') as mock_getsize,
        ):
            # Simulate large file (1GB)
            mock_getsize.return_value = 1024 * 1024 * 1024

            mock_instance = Mock()
            mock_artifact_version.return_value = mock_instance
            mock_instance.upload.return_value = Mock(id="large_file_version")

            result = self.runner.invoke(
                artifact,
                [
                    'cp',
                    './large_file.txt',
                    'users/test_user/apps/test_app/artifacts/test_artifact',
                ],
                obj=mock_obj,
            )

            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_download_to_existing_file_no_overwrite(self, mock_validate):
        """Test download when target file exists and overwrite not forced."""
        mock_obj = setup_context_mock(mock_validate)

        with (
            patch('clarifai.client.artifact_version.ArtifactVersion') as mock_artifact_version,
            patch('os.path.exists') as mock_exists,
        ):
            # Simulate target file exists
            mock_exists.return_value = True

            mock_instance = Mock()
            mock_artifact_version.return_value = mock_instance
            # Simulate the builder would raise an error about file existing
            mock_instance.download.side_effect = UserError("File already exists")

            result = self.runner.invoke(
                artifact,
                [
                    'cp',
                    'users/test_user/apps/test_app/artifacts/test_artifact/versions/test_version',
                    './existing_file.txt',
                ],
                obj=mock_obj,
            )

            assert result.exit_code != 0
            assert "File already exists" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    def test_version_commands_with_special_characters(self, mock_validate):
        """Test version commands with special characters in IDs."""
        mock_obj = setup_context_mock(mock_validate)

        with patch('clarifai.client.artifact_version.ArtifactVersion') as mock_artifact_version:
            mock_instance = Mock()
            mock_artifact_version.return_value = mock_instance
            mock_instance.get.return_value = Mock(
                id='v1.0.0-alpha.1',
                artifact_id='my-model-v2',
                description='Test version',
                visibility=Mock(gettable=50),  # PUBLIC enum value
                expires_at=None,  # Never expires
                created_at=Mock(ToDatetime=Mock(return_value='2023-01-01 00:00:00')),
                modified_at=Mock(ToDatetime=Mock(return_value='2023-01-01 00:00:00')),
                upload=Mock(
                    id='upload123',
                    content_name='model.pkl',
                    content_length=1024,
                    status=Mock(description='UPLOADED'),
                ),
            )

            result = self.runner.invoke(
                artifact,
                [
                    'get',
                    'users/test_user/apps/test_app/artifacts/my-model-v2/versions/v1.0.0-alpha.1',
                ],
                obj=mock_obj,
            )

            assert result.exit_code == 0
            assert 'v1.0.0-alpha.1' in result.output


class TestArtifactVersionCLIIntegration:
    """Integration tests for artifact version CLI workflow."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    @patch('clarifai.cli.artifact.validate_context')
    @patch('os.path.exists')
    def test_complete_version_lifecycle(self, mock_exists, mock_validate):
        """Test complete version lifecycle - upload, list, get, delete."""
        mock_obj = setup_context_mock(mock_validate)
        mock_exists.return_value = True

        with patch('clarifai.client.artifact_version.ArtifactVersion') as mock_artifact_version:
            mock_instance = Mock()
            mock_artifact_version.return_value = mock_instance

            # 1. Upload new version
            mock_instance.upload.return_value = Mock(id="new_version")
            result = self.runner.invoke(
                artifact,
                ['cp', './test_file.txt', 'users/test_user/apps/test_app/artifacts/test_artifact'],
                obj=mock_obj,
            )
            assert result.exit_code == 0

            # 2. List versions
            mock_instance.list.return_value = [
                Mock(
                    id='new_version',
                    description='New version',
                    visibility=Mock(gettable=50),  # PUBLIC enum value
                    expires_at=None,  # Never expires
                    created_at=Mock(ToDatetime=Mock(return_value='2023-01-01 00:00:00')),
                )
            ]
            result = self.runner.invoke(
                artifact,
                ['list', 'users/test_user/apps/test_app/artifacts/test_artifact', '--versions'],
                obj=mock_obj,
            )
            assert result.exit_code == 0

            # 3. Get version info
            mock_instance.get.return_value = Mock(
                id='new_version',
                artifact_id='test_artifact',
                description='New version description',
                visibility=Mock(gettable=50),  # PUBLIC enum value
                expires_at=None,  # Never expires
                created_at=Mock(ToDatetime=Mock(return_value='2023-01-01 00:00:00')),
                modified_at=Mock(ToDatetime=Mock(return_value='2023-01-01 00:00:00')),
                upload=Mock(
                    id='upload123',
                    content_name='test_file.txt',
                    content_length=1024,
                    status=Mock(description='UPLOADED'),
                ),
            )
            result = self.runner.invoke(
                artifact,
                [
                    'get',
                    'users/test_user/apps/test_app/artifacts/test_artifact/versions/new_version',
                ],
                obj=mock_obj,
            )
            assert result.exit_code == 0

            # 4. Delete version
            mock_instance.delete.return_value = True
            result = self.runner.invoke(
                artifact,
                [
                    'delete',
                    'users/test_user/apps/test_app/artifacts/test_artifact/versions/new_version',
                ],
                input='y\n',
                obj=mock_obj,
            )
            assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__])
