"""Test file for artifact CLI functionality."""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from clarifai.cli.artifact import artifact, is_local_path
from clarifai.errors import UserError
from clarifai.runners.artifacts.artifact_builder import parse_artifact_path


class TestArtifactPathParsing:
    """Test class for artifact path parsing."""

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
        ]

        for invalid_path in invalid_paths:
            with pytest.raises(UserError, match="Invalid artifact path format"):
                parse_artifact_path(invalid_path)

    def test_is_local_path(self):
        """Test local path detection."""
        assert is_local_path("./local/file.txt") is True
        assert is_local_path("/home/user/file.txt") is True
        assert is_local_path("file.txt") is True
        assert is_local_path("users/u123/apps/a456/artifacts/my_artifact") is False


class TestArtifactCLI:
    """Test class for artifact CLI commands."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    @patch('clarifai.cli.artifact.validate_context')
    def test_list_command_success(self, mock_validate):
        """Test successful list command."""
        mock_validate.return_value = None

        # Set up mock context object
        mock_current = Mock()
        mock_current.to_grpc.return_value = {}
        mock_obj = Mock()
        mock_obj.current = mock_current

        with patch('clarifai.client.artifact.Artifact.list') as mock_list:
            mock_artifact1 = Mock()
            mock_artifact1.artifact_id = 'artifact1'
            mock_artifact1.info.return_value = {
                'user_id': 'test_user',
                'app_id': 'test_app',
                'created_at': '2024-01-01',
            }

            mock_artifact2 = Mock()
            mock_artifact2.artifact_id = 'artifact2'
            mock_artifact2.info.return_value = {
                'user_id': 'test_user',
                'app_id': 'test_app',
                'created_at': '2024-01-01',
            }

            mock_list.return_value = [mock_artifact1, mock_artifact2]

            result = self.runner.invoke(
                artifact, ['list', 'users/test_user/apps/test_app'], obj=mock_obj
            )

            assert result.exit_code == 0
            assert 'artifact1' in result.output

    @patch('clarifai.cli.artifact.validate_context')
    def test_list_command_missing_params(self, mock_validate):
        """Test list command with missing required parameters."""
        mock_validate.return_value = None

        result = self.runner.invoke(artifact, ['list'])
        assert result.exit_code != 0
        assert "user_id and app_id are required" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    def test_list_versions_command(self, mock_validate):
        """Test list versions command."""
        mock_validate.return_value = None

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.list_artifact_versions.return_value = [
                {'id': 'version1', 'artifact_id': 'test_artifact'},
                {'id': 'version2', 'artifact_id': 'test_artifact'},
            ]

            result = self.runner.invoke(
                artifact,
                ['list', 'users/test_user/apps/test_app/artifacts/test_artifact', '--versions'],
            )

            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_get_command_success(self, mock_validate):
        """Test successful get command."""
        mock_validate.return_value = None

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.get_artifact_info.return_value = {
                'id': 'test_artifact',
                'description': 'Test artifact',
                'created_at': '2023-01-01T00:00:00Z',
            }

            result = self.runner.invoke(
                artifact, ['get', 'users/test_user/apps/test_app/artifacts/test_artifact']
            )

            assert result.exit_code == 0
            assert 'test_artifact' in result.output

    @patch('clarifai.cli.artifact.validate_context')
    def test_get_command_missing_params(self, mock_validate):
        """Test get command with missing required parameters."""
        mock_validate.return_value = None

        result = self.runner.invoke(artifact, ['get', 'incomplete/path'])
        assert result.exit_code != 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_delete_command_success(self, mock_validate):
        """Test successful delete command."""
        mock_validate.return_value = None

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.delete_artifact.return_value = True

            # Use input to simulate user confirmation
            result = self.runner.invoke(
                artifact,
                ['delete', 'users/test_user/apps/test_app/artifacts/test_artifact'],
                input='y\n',
            )

            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_delete_command_cancel(self, mock_validate):
        """Test delete command with user cancellation."""
        mock_validate.return_value = None

        result = self.runner.invoke(
            artifact,
            ['delete', 'users/test_user/apps/test_app/artifacts/test_artifact'],
            input='n\n',
        )

        assert result.exit_code == 0
        assert "Operation cancelled" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    @patch('os.path.exists')
    def test_cp_command_upload_success(self, mock_exists, mock_validate):
        """Test successful upload via cp command."""
        mock_validate.return_value = None
        mock_exists.return_value = True

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.upload_from_path.return_value = Mock(id="uploaded_version")

            result = self.runner.invoke(
                artifact,
                ['cp', './test_file.txt', 'users/test_user/apps/test_app/artifacts/test_artifact'],
            )

            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_cp_command_download_success(self, mock_validate):
        """Test successful download via cp command."""
        mock_validate.return_value = None

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            mock_instance.download_from_path.return_value = "./downloaded_file.txt"

            result = self.runner.invoke(
                artifact,
                [
                    'cp',
                    'users/test_user/apps/test_app/artifacts/test_artifact',
                    './downloaded_file.txt',
                ],
            )

            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_cp_command_invalid_paths(self, mock_validate):
        """Test cp command with invalid path combinations."""
        mock_validate.return_value = None

        # Both paths are local
        result = self.runner.invoke(artifact, ['cp', './local1.txt', './local2.txt'])
        assert result.exit_code != 0
        assert (
            "One of source or destination must be a local path and the other an artifact path"
            in result.output
        )

        # Both paths are remote
        result = self.runner.invoke(
            artifact, ['cp', 'users/u1/apps/a1/artifacts/art1', 'users/u2/apps/a2/artifacts/art2']
        )
        assert result.exit_code != 0
        assert (
            "One of source or destination must be a local path and the other an artifact path"
            in result.output
        )

    @patch('clarifai.cli.artifact.validate_context')
    @patch('os.path.exists')
    def test_cp_command_missing_file(self, mock_exists, mock_validate):
        """Test cp command with missing local file."""
        mock_validate.return_value = None
        mock_exists.return_value = False

        result = self.runner.invoke(
            artifact,
            [
                'cp',
                './nonexistent_file.txt',
                'users/test_user/apps/test_app/artifacts/test_artifact',
            ],
        )

        assert result.exit_code != 0
        assert "Error uploading file:" in result.output
        assert "Source file does not exist" in result.output

    def test_artifact_alias_commands(self):
        """Test artifact CLI aliases work correctly."""
        # Test that 'af' alias works
        result = self.runner.invoke(artifact, ['--help'])
        assert result.exit_code == 0
        assert "Manage Artifacts" in result.output

        # Test that 'ls' alias works for list command
        with patch('clarifai.cli.artifact.validate_context'):
            result = self.runner.invoke(artifact, ['ls', '--help'])
            assert result.exit_code == 0


class TestArtifactCLIIntegration:
    """Integration tests for artifact CLI commands."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    @patch('clarifai.cli.artifact.validate_context')
    def test_full_workflow_simulation(self, mock_validate):
        """Test simulated full workflow - list, get, delete."""
        mock_validate.return_value = None

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance

            # Mock list response
            mock_instance.list_artifacts.return_value = [
                {'id': 'test_artifact', 'description': 'Test artifact'}
            ]

            # Test list
            result = self.runner.invoke(artifact, ['list', 'users/test_user/apps/test_app'])
            assert result.exit_code == 0

            # Mock get response
            mock_instance.get_artifact_info.return_value = {
                'id': 'test_artifact',
                'description': 'Test artifact',
                'created_at': '2023-01-01T00:00:00Z',
            }

            # Test get
            result = self.runner.invoke(
                artifact, ['get', 'users/test_user/apps/test_app/artifacts/test_artifact']
            )
            assert result.exit_code == 0

            # Mock delete response
            mock_instance.delete_artifact.return_value = True

            # Test delete with confirmation
            result = self.runner.invoke(
                artifact,
                ['delete', 'users/test_user/apps/test_app/artifacts/test_artifact'],
                input='y\n',
            )
            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_error_handling(self, mock_validate):
        """Test CLI error handling."""
        mock_validate.return_value = None

        with patch('clarifai.runners.artifacts.artifact_builder.ArtifactBuilder') as mock_builder:
            mock_instance = Mock()
            mock_builder.return_value = mock_instance

            # Simulate an error
            mock_instance.get_artifact_info.side_effect = UserError("Artifact not found")

            result = self.runner.invoke(
                artifact, ['get', 'users/test_user/apps/test_app/artifacts/nonexistent']
            )

            assert result.exit_code != 0
            assert "Error getting artifact information:" in result.output
            assert "Artifact not found" in result.output


if __name__ == "__main__":
    pytest.main([__file__])
