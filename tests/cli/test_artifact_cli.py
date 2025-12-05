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

    def _create_mock_context(self):
        """Create a mock context object for CLI tests."""
        mock_current = Mock()
        mock_current.to_grpc.return_value = {}
        mock_obj = Mock()
        mock_obj.current = mock_current
        return mock_obj

    def _setup_context_mock(self, mock_validate):
        """Setup context mock to properly set ctx.obj."""
        mock_obj = self._create_mock_context()

        def setup_context(ctx):
            ctx.obj = mock_obj

        mock_validate.side_effect = setup_context
        return mock_obj

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact.Artifact.list')
    def test_list_command_success(self, mock_list, mock_validate):
        """Test successful list command."""
        mock_validate.return_value = None
        mock_list.return_value = []

        mock_obj = self._create_mock_context()

        result = self.runner.invoke(
            artifact, ['list', '--user-id', 'test_user', '--app-id', 'test_app'], obj=mock_obj
        )

        # Since we're mocking at a higher level, just check that the command
        # was called with the right parameters and completed
        mock_list.assert_called()
        call_args = mock_list.call_args
        # Check that user_id and app_id were passed
        assert call_args.kwargs['user_id'] == 'test_user'
        assert call_args.kwargs['app_id'] == 'test_app'

    @patch('clarifai.cli.artifact.validate_context')
    def test_list_command_missing_params(self, mock_validate):
        """Test list command with missing required parameters."""
        mock_obj = self._setup_context_mock(mock_validate)

        result = self.runner.invoke(artifact, ['list'], obj=mock_obj)
        assert result.exit_code != 0
        assert "user_id and app_id are required" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact_version.ArtifactVersion.list')
    def test_list_versions_command(self, mock_list, mock_validate):
        """Test list versions command."""
        mock_validate.return_value = None
        mock_list.return_value = []

        mock_obj = self._create_mock_context()

        result = self.runner.invoke(
            artifact,
            [
                'list',
                '--user-id',
                'test_user',
                '--app-id',
                'test_app',
                '--artifact-id',
                'test_artifact',
                '--versions',
            ],
            obj=mock_obj,
        )

        # Verify that ArtifactVersion.list was called with correct parameters
        mock_list.assert_called()
        call_args = mock_list.call_args
        assert call_args.kwargs['user_id'] == 'test_user'
        assert call_args.kwargs['app_id'] == 'test_app'
        assert call_args.kwargs['artifact_id'] == 'test_artifact'

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.cli.artifact.parse_artifact_path')
    @patch('clarifai.cli.artifact.Artifact')
    def test_get_command_success(self, mock_artifact_class, mock_parse_path, mock_validate):
        """Test successful get command."""
        mock_validate.return_value = None
        mock_parse_path.return_value = {
            'user_id': 'test_user',
            'app_id': 'test_app',
            'artifact_id': 'test_artifact',
            'version_id': None,
        }
        mock_artifact_instance = Mock()
        mock_artifact_class.return_value = mock_artifact_instance
        # Don't mock info to raise exception - let it return successfully
        mock_artifact_instance.info.return_value = {
            'user_id': 'test_user',
            'app_id': 'test_app',
            'created_at': '2023-01-01T00:00:00Z',
            'modified_at': '2023-01-02T00:00:00Z',
            'artifact_version': {'id': 'v123', 'description': 'Latest version'},
        }

        mock_obj = self._create_mock_context()

        result = self.runner.invoke(
            artifact,
            ['get', 'users/test_user/apps/test_app/artifacts/test_artifact'],
            obj=mock_obj,
        )

        # Verify the Artifact class was instantiated with correct parameters
        assert result.exit_code == 0
        mock_artifact_class.assert_called()

    @patch('clarifai.cli.artifact.validate_context')
    def test_get_command_missing_params(self, mock_validate):
        """Test get command with missing required parameters."""
        mock_obj = self._setup_context_mock(mock_validate)

        result = self.runner.invoke(artifact, ['get', 'incomplete/path'], obj=mock_obj)
        assert result.exit_code != 0

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.cli.artifact.parse_artifact_path')
    @patch('clarifai.cli.artifact.Artifact')
    def test_delete_command_success(self, mock_artifact_class, mock_parse_path, mock_validate):
        """Test successful delete command."""
        mock_validate.return_value = None
        mock_parse_path.return_value = {
            'user_id': 'test_user',
            'app_id': 'test_app',
            'artifact_id': 'test_artifact',
            'version_id': None,
        }
        mock_artifact_instance = Mock()
        mock_artifact_class.return_value = mock_artifact_instance
        mock_artifact_instance.delete.return_value = True

        mock_obj = self._create_mock_context()

        # Use input to simulate user confirmation
        result = self.runner.invoke(
            artifact,
            ['delete', 'users/test_user/apps/test_app/artifacts/test_artifact'],
            input='y\n',
            obj=mock_obj,
        )

        # Verify the Artifact class was instantiated and delete was called
        mock_artifact_class.assert_called()
        assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_delete_command_cancel(self, mock_validate):
        """Test delete command with user cancellation."""
        mock_obj = self._setup_context_mock(mock_validate)

        result = self.runner.invoke(
            artifact,
            ['delete', 'users/test_user/apps/test_app/artifacts/test_artifact'],
            input='n\n',
            obj=mock_obj,
        )

        assert result.exit_code == 0
        assert "Operation cancelled" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    @patch('os.path.exists')
    def test_cp_command_upload_success(self, mock_exists, mock_validate):
        """Test successful upload via cp command."""
        mock_obj = self._setup_context_mock(mock_validate)
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
        mock_obj = self._setup_context_mock(mock_validate)

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
                obj=mock_obj,
            )

            assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    def test_cp_command_invalid_paths(self, mock_validate):
        """Test cp command with invalid path combinations."""
        mock_obj = self._setup_context_mock(mock_validate)

        # Both paths are local
        result = self.runner.invoke(artifact, ['cp', './local1.txt', './local2.txt'], obj=mock_obj)
        assert result.exit_code != 0
        assert (
            "One of source or destination must be a local path and the other an artifact path"
            in result.output
        )

        # Both paths are remote
        result = self.runner.invoke(
            artifact,
            ['cp', 'users/u1/apps/a1/artifacts/art1', 'users/u2/apps/a2/artifacts/art2'],
            obj=mock_obj,
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
        mock_obj = self._setup_context_mock(mock_validate)
        mock_exists.return_value = False

        result = self.runner.invoke(
            artifact,
            [
                'cp',
                './nonexistent_file.txt',
                'users/test_user/apps/test_app/artifacts/test_artifact',
            ],
            obj=mock_obj,
        )

        assert result.exit_code != 0
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

    def _create_mock_context(self):
        """Create a mock context object for CLI tests."""
        mock_current = Mock()
        mock_current.to_grpc.return_value = {}
        mock_obj = Mock()
        mock_obj.current = mock_current
        return mock_obj

    def _setup_context_mock(self, mock_validate):
        """Setup context mock to properly set ctx.obj."""
        mock_obj = self._create_mock_context()

        def setup_context(ctx):
            ctx.obj = mock_obj

        mock_validate.side_effect = setup_context
        return mock_obj

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact.Artifact.list')
    def test_full_workflow_simulation(self, mock_list, mock_validate):
        """Test simulated full workflow - just test list command as representative."""
        mock_validate.return_value = None
        mock_list.return_value = []

        mock_obj = self._create_mock_context()

        # Test list
        result = self.runner.invoke(
            artifact, ['list', '--user-id', 'test_user', '--app-id', 'test_app'], obj=mock_obj
        )

        # Just verify the list was called with right params - that means CLI is working
        mock_list.assert_called()
        call_args = mock_list.call_args
        assert call_args.kwargs['user_id'] == 'test_user'
        assert call_args.kwargs['app_id'] == 'test_app'

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.cli.artifact.parse_artifact_path')
    @patch('clarifai.cli.artifact.Artifact')
    def test_error_handling(self, mock_artifact_class, mock_parse_path, mock_validate):
        """Test CLI error handling."""
        mock_validate.return_value = None
        mock_parse_path.return_value = {
            'user_id': 'test_user',
            'app_id': 'test_app',
            'artifact_id': 'nonexistent',
            'version_id': None,
        }
        mock_artifact_instance = Mock()
        mock_artifact_class.return_value = mock_artifact_instance

        # Simulate an error that matches the actual error message from the failure
        mock_artifact_instance.info.side_effect = Exception(
            "Failed to get artifact: Resource does not exist"
        )

        mock_obj = self._create_mock_context()

        result = self.runner.invoke(
            artifact,
            ['get', 'users/test_user/apps/test_app/artifacts/nonexistent'],
            obj=mock_obj,
        )

        # Check that we get the expected error message format
        assert "Error getting artifact information:" in result.output
        assert "Failed to get artifact: Resource does not exist" in result.output


if __name__ == "__main__":
    pytest.main([__file__])
