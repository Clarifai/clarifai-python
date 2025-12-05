"""Test file for artifact version CLI functionality."""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from clarifai.cli.artifact import artifact
from clarifai.errors import UserError


class TestArtifactVersionCLI:
    """Test class for artifact version CLI commands."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    def _create_mock_context(self):
        """Create a mock context object for CLI tests."""
        mock_current = Mock()
        mock_current.to_grpc.return_value = {}
        mock_obj = Mock()
        mock_obj.current = mock_current
        return mock_obj

    @patch('clarifai.cli.artifact.validate_context')
    def test_list_versions_command_success(self, mock_validate):
        """Test successful list versions command."""
        mock_validate.return_value = None

        mock_obj = self._create_mock_context()

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.list_artifact_versions.return_value = [
                {
                    'id': 'version1',
                    'artifact_id': 'test_artifact',
                    'created_at': '2023-01-01T00:00:00Z',
                    'size': 1024,
                },
                {
                    'id': 'version2',
                    'artifact_id': 'test_artifact',
                    'created_at': '2023-01-02T00:00:00Z',
                    'size': 2048,
                },
            ]

            result = self.runner.invoke(
                artifact,
                ['list', 'users/test_user/apps/test_app/artifacts/test_artifact', '--versions'],
                obj=mock_obj
            )

            assert result.exit_code == 0
            assert 'version1' in result.output
            assert 'version2' in result.output

    @patch('clarifai.cli.artifact.validate_context')
    def test_get_version_command_success(self, mock_validate):
        """Test successful get version command."""
        mock_validate.return_value = None

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.get_artifact_version_info.return_value = {
                'id': 'test_version',
                'artifact_id': 'test_artifact',
                'created_at': '2023-01-01T00:00:00Z',
                'size': 1024,
                'description': 'Test version',
            }

            result = self.runner.invoke(
                artifact,
                [
                    'get',
                    'users/test_user/apps/test_app/artifacts/test_artifact/versions/test_version',
                ],
            )

            assert result.exit_code == 0
            assert 'test_version' in result.output
            assert 'test_artifact' in result.output

    @patch('clarifai.cli.artifact.validate_context')
    def test_delete_version_command_success(self, mock_validate):
        """Test successful delete version command."""
        mock_validate.return_value = None

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.delete_artifact_version.return_value = True

            result = self.runner.invoke(
                artifact,
                [
                    'delete',
                    'users/test_user/apps/test_app/artifacts/test_artifact/versions/test_version',
                ],
                input='y\n',
            )

            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_delete_version_command_cancel(self, mock_validate):
        """Test delete version command with user cancellation."""
        mock_validate.return_value = None

        result = self.runner.invoke(
            artifact,
            [
                'delete',
                'users/test_user/apps/test_app/artifacts/test_artifact/versions/test_version',
            ],
            input='n\n',
        )

        assert result.exit_code == 0
        assert "Operation cancelled" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    @patch('os.path.exists')
    def test_cp_upload_to_version_success(self, mock_exists, mock_validate):
        """Test successful upload to specific version via cp command."""
        mock_validate.return_value = None
        mock_exists.return_value = True

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.upload_from_path.return_value = Mock(id="new_version")

            result = self.runner.invoke(
                artifact,
                [
                    'cp',
                    './test_file.txt',
                    'users/test_user/apps/test_app/artifacts/test_artifact',
                    '--description',
                    'New version upload',
                ],
            )

            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_cp_download_specific_version_success(self, mock_validate):
        """Test successful download of specific version via cp command."""
        mock_validate.return_value = None

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.download_from_path.return_value = "./downloaded_version.txt"

            result = self.runner.invoke(
                artifact,
                [
                    'cp',
                    'users/test_user/apps/test_app/artifacts/test_artifact/versions/test_version',
                    './downloaded_version.txt',
                ],
            )

            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_invalid_version_path_handling(self, mock_validate):
        """Test handling of invalid version paths."""
        mock_validate.return_value = None

        # Test invalid version path format
        result = self.runner.invoke(
            artifact,
            ['get', 'users/test_user/apps/test_app/artifacts/test_artifact/invalid_versions_path'],
        )

        assert result.exit_code != 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_version_error_handling(self, mock_validate):
        """Test version-specific error handling."""
        mock_validate.return_value = None

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance

            # Simulate version not found error
            mock_instance.get_artifact_version_info.side_effect = UserError("Version not found")

            result = self.runner.invoke(
                artifact,
                [
                    'get',
                    'users/test_user/apps/test_app/artifacts/test_artifact/versions/nonexistent',
                ],
            )

            assert result.exit_code != 0
            assert "Error getting artifact information:" in result.output
            assert "Version not found" in result.output


class TestArtifactVersionCLIEdgeCases:
    """Test edge cases for artifact version CLI commands."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    @patch('clarifai.cli.artifact.validate_context')
    def test_list_versions_empty_result(self, mock_validate):
        """Test list versions with empty result."""
        mock_validate.return_value = None

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.list_artifact_versions.return_value = []

            result = self.runner.invoke(
                artifact,
                ['list', 'users/test_user/apps/test_app/artifacts/test_artifact', '--versions'],
            )

            assert result.exit_code == 0
            assert "No versions found" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    def test_get_version_missing_version_id(self, mock_validate):
        """Test get command with missing version ID in path."""
        mock_validate.return_value = None

        # Path that ends with '/versions/' but no version ID
        result = self.runner.invoke(
            artifact, ['get', 'users/test_user/apps/test_app/artifacts/test_artifact/versions/']
        )

        assert result.exit_code != 0

    @patch('clarifai.cli.artifact.validate_context')
    @patch('os.path.exists')
    def test_upload_large_file_simulation(self, mock_exists, mock_validate):
        """Test upload simulation for large files."""
        mock_validate.return_value = None
        mock_exists.return_value = True

        with (
            patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder,
            patch('os.path.getsize') as mock_getsize,
        ):
            # Simulate large file (1GB)
            mock_getsize.return_value = 1024 * 1024 * 1024

            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.upload_from_path.return_value = Mock(id="large_file_version")

            result = self.runner.invoke(
                artifact,
                [
                    'cp',
                    './large_file.txt',
                    'users/test_user/apps/test_app/artifacts/test_artifact',
                ],
            )

            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_download_to_existing_file_no_overwrite(self, mock_validate):
        """Test download when target file exists and overwrite not forced."""
        mock_validate.return_value = None

        with (
            patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder,
            patch('os.path.exists') as mock_exists,
        ):
            # Simulate target file exists
            mock_exists.return_value = True

            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            # Simulate the builder would raise an error about file existing
            mock_instance.download_from_path.side_effect = UserError("File already exists")

            result = self.runner.invoke(
                artifact,
                [
                    'cp',
                    'users/test_user/apps/test_app/artifacts/test_artifact/versions/test_version',
                    './existing_file.txt',
                ],
            )

            assert result.exit_code != 0
            assert "Error downloading file:" in result.output
            assert "File already exists" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    def test_version_commands_with_special_characters(self, mock_validate):
        """Test version commands with special characters in IDs."""
        mock_validate.return_value = None

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.get_artifact_version_info.return_value = {
                'id': 'v1.0.0-alpha.1',
                'artifact_id': 'my-model-v2',
                'created_at': '2023-01-01T00:00:00Z',
                'size': 1024,
            }

            result = self.runner.invoke(
                artifact,
                [
                    'get',
                    'users/test_user/apps/test_app/artifacts/my-model-v2/versions/v1.0.0-alpha.1',
                ],
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
        mock_validate.return_value = None
        mock_exists.return_value = True

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance

            # 1. Upload new version
            mock_instance.upload_from_path.return_value = Mock(id="new_version")
            result = self.runner.invoke(
                artifact,
                ['cp', './test_file.txt', 'users/test_user/apps/test_app/artifacts/test_artifact'],
            )
            assert result.exit_code == 0

            # 2. List versions
            mock_instance.list_artifact_versions.return_value = [
                {'id': 'new_version', 'artifact_id': 'test_artifact'}
            ]
            result = self.runner.invoke(
                artifact,
                ['list', 'users/test_user/apps/test_app/artifacts/test_artifact', '--versions'],
            )
            assert result.exit_code == 0

            # 3. Get version info
            mock_instance.get_artifact_version_info.return_value = {
                'id': 'new_version',
                'artifact_id': 'test_artifact',
                'created_at': '2023-01-01T00:00:00Z',
            }
            result = self.runner.invoke(
                artifact,
                [
                    'get',
                    'users/test_user/apps/test_app/artifacts/test_artifact/versions/new_version',
                ],
            )
            assert result.exit_code == 0

            # 4. Delete version
            mock_instance.delete_artifact_version.return_value = True
            result = self.runner.invoke(
                artifact,
                [
                    'delete',
                    'users/test_user/apps/test_app/artifacts/test_artifact/versions/new_version',
                ],
                input='y\n',
            )
            assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__])
