"""Test file for artifact CLI functionality."""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from clarifai.cli.artifact import (
    _download_artifact,
    _upload_artifact,
    artifact,
    is_local_path,
    parse_artifact_path,
)
from clarifai.errors import UserError
from tests.test_artifact_utils import create_mock_context, setup_context_mock


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

    def test_parse_app_level_path(self):
        """Test parsing app-level paths for auto-creation."""
        path = "users/u123/apps/a456"
        parsed = parse_artifact_path(path)

        assert parsed['user_id'] == 'u123'
        assert parsed['app_id'] == 'a456'
        assert parsed['artifact_id'] is None
        assert parsed['version_id'] is None

    def test_is_local_path(self):
        """Test local path detection."""
        # Positive cases (local paths)
        assert is_local_path("./local/file.txt") is True
        assert is_local_path("/home/user/file.txt") is True
        assert is_local_path("file.txt") is True
        assert is_local_path("../parent/file.txt") is True
        assert is_local_path("~/home/file.txt") is True

        # Malformed artifact-like paths should be treated as local
        assert is_local_path("users/u123/apps/a456/artifacts") is True  # Incomplete
        assert is_local_path("users/u123/apps/a456/other") is True  # Wrong structure
        assert (
            is_local_path("users/u123/apps/a456/artifacts/my_artifact/other") is True
        )  # Invalid suffix

        # Negative cases (artifact paths)
        assert is_local_path("users/u123/apps/a456/artifacts/my_artifact") is False
        assert is_local_path("users/u123/apps/a456") is False  # App-level path
        assert (
            is_local_path("users/u123/apps/a456/artifacts/my_artifact/versions/v1") is False
        )  # Version-level path


class TestArtifactCLI:
    """Test class for artifact CLI commands."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact.Artifact')
    def test_list_command_success(self, mock_artifact_class, mock_validate):
        """Test successful list command."""
        mock_validate.return_value = None

        # Mock the artifact instance and its list method
        mock_artifact_instance = Mock()
        mock_artifact_instance.list.return_value = []
        mock_artifact_class.return_value = mock_artifact_instance

        mock_obj = create_mock_context()

        result = self.runner.invoke(
            artifact, ['list', 'users/test_user/apps/test_app'], obj=mock_obj
        )

        if result.exit_code != 0:
            print(f"Command failed with output: {result.output}")

        # Check that the artifact was instantiated and list was called
        mock_artifact_class.assert_called_once()
        mock_artifact_instance.list.assert_called_once()
        assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.cli.artifact.display_co_resources')
    @patch('clarifai.client.artifact.Artifact')
    def test_list_command_latest_version_and_visibility(
        self, mock_artifact_class, mock_display_co_resources, mock_validate
    ):
        """Test artifact listing resolves latest version and visibility columns."""
        mock_validate.return_value = None

        mock_artifact_instance = Mock()
        mock_artifact = Mock()
        mock_artifact.id = 'test-artifact'
        mock_artifact.artifact_version = Mock()
        mock_artifact.artifact_version.id = ''
        mock_artifact.artifact_version.visibility.gettable = 10  # PRIVATE
        mock_artifact.artifact_version_id = 'v1'
        mock_artifact.created_at = None

        mock_artifact_instance.list.return_value = [mock_artifact]
        mock_artifact_class.return_value = mock_artifact_instance

        mock_obj = create_mock_context()

        result = self.runner.invoke(
            artifact, ['list', 'users/test_user/apps/test_app'], obj=mock_obj
        )

        assert result.exit_code == 0
        mock_display_co_resources.assert_called_once()

        custom_columns = mock_display_co_resources.call_args.kwargs['custom_columns']
        assert custom_columns['LATEST_VERSION'](mock_artifact) == 'v1'
        assert custom_columns['VISIBILITY'](mock_artifact) == 'PRIVATE'

    @patch('clarifai.cli.artifact.validate_context')
    def test_list_command_missing_params(self, mock_validate):
        """Test list command with missing required parameters."""
        mock_obj = setup_context_mock(mock_validate)

        result = self.runner.invoke(artifact, ['list'], obj=mock_obj)
        assert result.exit_code != 0
        assert "Missing argument 'PATH'" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact_version.ArtifactVersion')
    def test_list_versions_command(self, mock_artifact_version_class, mock_validate):
        """Test list versions command."""
        mock_validate.return_value = None

        # Mock the artifact version instance and its list method
        mock_version_instance = Mock()
        mock_version_instance.list.return_value = []
        mock_artifact_version_class.return_value = mock_version_instance

        mock_obj = create_mock_context()

        result = self.runner.invoke(
            artifact,
            [
                'list',
                'users/test_user/apps/test_app/artifacts/test_artifact',
                '--versions',
            ],
            obj=mock_obj,
        )

        # Verify that ArtifactVersion.list was called with correct parameters
        mock_artifact_version_class.assert_called_once()
        mock_version_instance.list.assert_called_once()
        assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.cli.artifact.parse_artifact_path')
    @patch('clarifai.client.artifact.Artifact')
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

        mock_obj = create_mock_context()

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
        mock_obj = setup_context_mock(mock_validate)

        result = self.runner.invoke(artifact, ['get', 'incomplete/path'], obj=mock_obj)
        assert result.exit_code != 0

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.cli.artifact.parse_artifact_path')
    @patch('clarifai.client.artifact.Artifact')
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

        mock_obj = create_mock_context()

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
        mock_obj = setup_context_mock(mock_validate)

        result = self.runner.invoke(
            artifact,
            ['delete', 'users/test_user/apps/test_app/artifacts/test_artifact'],
            input='n\n',
            obj=mock_obj,
        )

        assert result.exit_code == 0
        assert "Operation cancelled" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact.Artifact')
    def test_delete_command_force(self, mock_artifact_class, mock_validate):
        """Test delete command with force flag (no confirmation needed)."""
        mock_obj = setup_context_mock(mock_validate)

        # Mock the artifact instance and its delete method
        mock_artifact_instance = Mock()
        mock_artifact_instance.delete.return_value = True
        mock_artifact_class.return_value = mock_artifact_instance

        result = self.runner.invoke(
            artifact,
            ['delete', 'users/test_user/apps/test_app/artifacts/test_artifact', '--force'],
            obj=mock_obj,
        )

        assert result.exit_code == 0
        assert "Successfully deleted artifact test_artifact" in result.output
        # Should not contain confirmation prompt or cancellation message
        assert "Are you sure" not in result.output
        assert "Operation cancelled" not in result.output
        mock_artifact_instance.delete.assert_called_once()

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact_version.ArtifactVersion')
    def test_delete_version_command_force(self, mock_artifact_version_class, mock_validate):
        """Test delete version command with force flag (no confirmation needed)."""
        mock_obj = setup_context_mock(mock_validate)

        # Mock the artifact version instance and its delete method
        mock_version_instance = Mock()
        mock_version_instance.delete.return_value = True
        mock_artifact_version_class.return_value = mock_version_instance

        result = self.runner.invoke(
            artifact,
            [
                'delete',
                'users/test_user/apps/test_app/artifacts/test_artifact/versions/v123',
                '--force',
            ],
            obj=mock_obj,
        )

        assert result.exit_code == 0
        assert "Successfully deleted artifact version v123" in result.output
        # Should not contain confirmation prompt or cancellation message
        assert "Are you sure" not in result.output
        assert "Operation cancelled" not in result.output
        mock_version_instance.delete.assert_called_once()

    @patch('clarifai.cli.artifact.validate_context')
    @patch('os.path.exists')
    @patch('clarifai.client.artifact_version.ArtifactVersion.upload')
    def test_cp_command_upload_success(self, mock_upload, mock_exists, mock_validate):
        """Test successful upload via cp command."""
        mock_obj = setup_context_mock(mock_validate)
        mock_exists.return_value = True

        # Mock successful upload
        mock_version = Mock()
        mock_version.artifact_id = "test_artifact"
        mock_version.version_id = "uploaded_version"
        mock_upload.return_value = mock_version

        result = self.runner.invoke(
            artifact,
            ['cp', './test_file.txt', 'users/test_user/apps/test_app/artifacts/test_artifact'],
            obj=mock_obj,
        )

        if result.exit_code != 0:
            print(f"Command failed with output: {result.output}")
        assert result.exit_code == 0
        mock_upload.assert_called_once()

    @patch('clarifai.cli.artifact._upload_artifact')
    @patch('os.path.exists')
    @patch('clarifai.cli.artifact.validate_context')
    def test_cp_command_upload_org_visibility(self, mock_validate, mock_exists, mock_upload):
        """Test upload with org visibility option."""
        mock_obj = setup_context_mock(mock_validate)
        mock_exists.return_value = True

        # Mock successful upload
        mock_version = Mock()
        mock_version.artifact_id = "test_artifact"
        mock_version.version_id = "uploaded_version"
        mock_upload.return_value = mock_version

        result = self.runner.invoke(
            artifact,
            [
                'cp',
                './test_file.txt',
                'users/test_user/apps/test_app/artifacts/test_artifact',
                '--visibility',
                'org',
            ],
            obj=mock_obj,
        )

        if result.exit_code != 0:
            print(f"Command failed with output: {result.output}")
        assert result.exit_code == 0
        mock_upload.assert_called_once()

        # Check that the upload was called with org visibility
        args, kwargs = mock_upload.call_args
        assert kwargs.get('visibility') == 'org'

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact_version.ArtifactVersion')
    @patch('clarifai.client.artifact.Artifact')
    def test_cp_command_download_success(
        self, mock_artifact_class, mock_artifact_version_class, mock_validate
    ):
        """Test successful download via cp command."""
        mock_obj = setup_context_mock(mock_validate)

        # Mock the artifact to return latest version info
        mock_artifact_instance = Mock()
        mock_artifact_info = Mock()
        mock_artifact_info.artifact_version.id = "latest_version"
        mock_artifact_instance.get.return_value = mock_artifact_info
        mock_artifact_class.return_value = mock_artifact_instance

        # Mock the artifact version and download
        mock_version_instance = Mock()
        mock_version_instance.download.return_value = "./downloaded_file.txt"
        mock_artifact_version_class.return_value = mock_version_instance

        result = self.runner.invoke(
            artifact,
            ['cp', 'users/test_user/apps/test_app/artifacts/test_artifact', './local_dir/'],
            obj=mock_obj,
        )

        if result.exit_code != 0:
            print(f"Command failed with output: {result.output}")
        assert result.exit_code == 0
        mock_version_instance.download.assert_called_once()

    @patch('clarifai.cli.artifact.validate_context')
    def test_cp_command_invalid_paths(self, mock_validate):
        """Test cp command with invalid path combinations."""
        mock_obj = setup_context_mock(mock_validate)

        # Both paths are local
        result = self.runner.invoke(artifact, ['cp', './local1.txt', './local2.txt'], obj=mock_obj)
        assert result.exit_code != 0
        assert "One of source or destination must be an artifact path" in result.output

        # Both paths are remote
        result = self.runner.invoke(
            artifact,
            ['cp', 'users/u1/apps/a1/artifacts/art1', 'users/u2/apps/a2/artifacts/art2'],
            obj=mock_obj,
        )
        assert result.exit_code != 0
        assert "One of source or destination must be a local path" in result.output

    @patch('clarifai.cli.artifact.validate_context')
    @patch('os.path.exists')
    def test_cp_command_missing_file(self, mock_exists, mock_validate):
        """Test cp command with missing local file."""
        mock_obj = setup_context_mock(mock_validate)
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

    def test_instance_reuse_across_operations(self):
        """Test that CLI operations properly handle instance reuse."""
        mock_obj = setup_context_mock(Mock())

        with patch('clarifai.client.artifact.Artifact') as mock_artifact:
            mock_instance = Mock()
            mock_artifact.return_value = mock_instance

            # Mock different operations
            mock_instance.create.return_value = Mock(artifact_id="test_artifact")
            mock_instance.get.return_value = Mock(
                id="test_artifact",
                description="Test artifact",
                artifact_version=Mock(id="v1"),
                created_at=Mock(ToDatetime=Mock(return_value="2023-01-01 00:00:00")),
                modified_at=Mock(ToDatetime=Mock(return_value="2023-01-01 00:00:00")),
            )

            mock_list_item = Mock(
                id="test_artifact",
                description="Test artifact",
                artifact_version_id="v1",
                created_at=Mock(ToDatetime=Mock(return_value="2023-01-01 00:00:00")),
            )
            mock_list_item.artifact_version = Mock(id="")
            mock_list_item.artifact_version.visibility.gettable = 10  # PRIVATE
            mock_instance.list.return_value = [mock_list_item]

            # Test multiple operations
            result1 = self.runner.invoke(
                artifact,
                ['get', 'users/test_user/apps/test_app/artifacts/test_artifact'],
                obj=mock_obj,
            )
            assert result1.exit_code == 0

            result2 = self.runner.invoke(
                artifact,
                ['list', 'users/test_user/apps/test_app'],
                obj=mock_obj,
            )
            assert result2.exit_code == 0

            # Verify both operations used proper instance initialization
            assert mock_artifact.call_count == 2

    def test_path_validation_edge_cases(self):
        """Test path validation with various edge cases."""
        # Test case 1: Normal valid path
        result = parse_artifact_path('users/user1/apps/app1/artifacts/artifact1')
        assert result['artifact_id'] == 'artifact1'

        # Test case 2: IDs with dots and hyphens
        result = parse_artifact_path('users/user-1.2/apps/app_v2.0/artifacts/model-v1.0.0')
        assert result['user_id'] == 'user-1.2'
        assert result['app_id'] == 'app_v2.0'
        assert result['artifact_id'] == 'model-v1.0.0'

    @patch('os.path.exists')
    @patch('clarifai.cli.artifact.validate_context')
    def test_cp_command_upload_app_level_error(self, mock_validate, mock_exists):
        """Test upload to app-level path. Should error since artifact_id is required."""
        mock_obj = setup_context_mock(mock_validate)
        mock_exists.return_value = True

        result = self.runner.invoke(
            artifact,
            ['cp', './test_file.txt', 'users/test_user/apps/test_app'],
            obj=mock_obj,
        )

        # Should fail because artifact_id is now required
        assert result.exit_code != 0
        assert "Path must include user_id, app_id, and artifact_id" in result.output


class TestArtifactCLIIntegration:
    """Integration tests for artifact CLI commands."""

    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact.Artifact')
    def test_full_workflow_simulation(self, mock_artifact_class, mock_validate):
        """Test simulated full workflow - just test list command as representative."""
        mock_validate.return_value = None

        # Mock the artifact instance and its list method
        mock_artifact_instance = Mock()
        mock_artifact_instance.list.return_value = []
        mock_artifact_class.return_value = mock_artifact_instance

        mock_obj = create_mock_context()

        # Test list
        result = self.runner.invoke(
            artifact, ['list', 'users/test_user/apps/test_app'], obj=mock_obj
        )

        # Just verify the list was called - that means CLI is working
        mock_artifact_instance.list.assert_called_once()
        assert result.exit_code == 0

    @patch('clarifai.cli.artifact.validate_context')
    @patch('clarifai.client.artifact.Artifact')
    def test_error_handling(self, mock_artifact_class, mock_validate):
        """Test CLI error handling."""
        mock_validate.return_value = None
        mock_artifact_instance = Mock()
        mock_artifact_class.return_value = mock_artifact_instance

        # Simulate an error in the get method
        mock_artifact_instance.get.side_effect = Exception(
            "Failed to get artifact: Resource does not exist"
        )

        mock_obj = create_mock_context()

        result = self.runner.invoke(
            artifact,
            ['get', 'users/test_user/apps/test_app/artifacts/nonexistent'],
            obj=mock_obj,
        )

        # Check that we get an error exit code
        assert result.exit_code != 0
        assert "Error getting artifact information:" in result.output


class TestConvenienceFunctions:
    """Test class for CLI convenience functions."""

    @patch('clarifai.client.artifact_version.ArtifactVersion')
    @patch('os.path.exists', return_value=True)
    def test_upload_artifact_function(self, mock_exists, mock_artifact_version_class):
        """Test _upload_artifact convenience function."""
        # Mock ArtifactVersion instance and its upload method
        mock_version_instance = Mock()
        mock_version_instance.upload.return_value = Mock(version_id="test_version")
        mock_artifact_version_class.return_value = mock_version_instance

        parsed_destination = {
            'user_id': 'u123',
            'app_id': 'a456',
            'artifact_id': 'my_artifact',
            'version_id': None,
        }
        client_kwargs = {'pat': 'test_pat', 'base': 'test_base'}

        result = _upload_artifact(
            source_path="./test_file.txt",
            parsed_destination=parsed_destination,
            client_kwargs=client_kwargs,
            description="Test upload",
            visibility="private",
        )

        # Verify instance was created with correct parameters
        mock_artifact_version_class.assert_called_once_with(
            artifact_id="my_artifact",
            version_id="",
            user_id="u123",
            app_id="a456",
            pat="test_pat",
            base="test_base",
        )

        # Verify upload method was called with correct parameters
        mock_version_instance.upload.assert_called_once_with(
            file_path="./test_file.txt",
            artifact_id="my_artifact",
            description="Test upload",
            visibility="private",
            expires_at=None,
            version_id=None,
        )

        assert result.version_id == "test_version"

    @patch('clarifai.client.artifact_version.ArtifactVersion')
    @patch('clarifai.client.artifact.Artifact')
    def test_download_artifact_function(self, mock_artifact_class, mock_artifact_version_class):
        """Test _download_artifact convenience function."""
        # Mock Artifact instance for getting latest version
        mock_artifact_instance = Mock()
        mock_artifact_info = Mock()
        mock_artifact_info.artifact_version.id = "latest_version"
        mock_artifact_instance.get.return_value = mock_artifact_info
        mock_artifact_class.return_value = mock_artifact_instance

        # Mock ArtifactVersion instance and its download method
        mock_version_instance = Mock()
        mock_version_instance.download.return_value = "/downloaded/path"
        mock_artifact_version_class.return_value = mock_version_instance

        parsed_source = {
            'user_id': 'u123',
            'app_id': 'a456',
            'artifact_id': 'my_artifact',
            'version_id': None,  # Download latest
        }
        client_kwargs = {'pat': 'test_pat', 'base': 'test_base'}

        result = _download_artifact(
            destination_path="./download/",
            parsed_source=parsed_source,
            client_kwargs=client_kwargs,
            force=True,
        )

        # Verify artifact was created to get latest version
        mock_artifact_class.assert_called_once_with(
            artifact_id="my_artifact",
            user_id="u123",
            app_id="a456",
            pat="test_pat",
            base="test_base",
        )

        # Verify artifact.get() was called
        mock_artifact_instance.get.assert_called_once()

        # Verify artifact version was created with latest version
        mock_artifact_version_class.assert_called_once_with(
            artifact_id="my_artifact",
            version_id="latest_version",
            user_id="u123",
            app_id="a456",
            pat="test_pat",
            base="test_base",
        )

        # Verify download was called
        mock_version_instance.download.assert_called_once_with(
            output_path="./download/", force=True
        )

        assert result == "/downloaded/path"

    @patch('clarifai.client.artifact_version.ArtifactVersion')
    def test_download_artifact_with_specific_version(self, mock_artifact_version_class):
        """Test _download_artifact with specific version ID."""
        # Mock ArtifactVersion instance and its download method
        mock_version_instance = Mock()
        mock_version_instance.download.return_value = "/downloaded/path"
        mock_artifact_version_class.return_value = mock_version_instance

        parsed_source = {
            'user_id': 'u123',
            'app_id': 'a456',
            'artifact_id': 'my_artifact',
            'version_id': 'v789',  # Specific version
        }
        client_kwargs = {'pat': 'test_pat'}

        result = _download_artifact(
            destination_path="./download/",
            parsed_source=parsed_source,
            client_kwargs=client_kwargs,
            force=False,
        )

        # Verify artifact version was created with specific version
        mock_artifact_version_class.assert_called_once_with(
            artifact_id="my_artifact",
            version_id="v789",
            user_id="u123",
            app_id="a456",
            pat="test_pat",
        )

        # Verify download was called
        mock_version_instance.download.assert_called_once_with(
            output_path="./download/", force=False
        )

        assert result == "/downloaded/path"


if __name__ == "__main__":
    pytest.main([__file__])
