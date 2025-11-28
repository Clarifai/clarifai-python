"""Test file for artifact CLI functionality."""

import pytest
from unittest.mock import Mock, patch
from click.testing import CliRunner

from clarifai.cli.artifact import artifact, parse_artifact_path, is_local_path
from clarifai.errors import UserError


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

    def test_list_command_missing_params(self):
        """Test list command with missing required parameters."""
        runner = CliRunner()
        
        with patch('clarifai.cli.artifact.validate_context'):
            result = runner.invoke(artifact, ['list'])
            assert result.exit_code != 0
            assert "user_id and app_id are required" in result.output

    def test_get_command_missing_params(self):
        """Test get command with missing required parameters."""
        runner = CliRunner()
        
        with patch('clarifai.cli.artifact.validate_context'):
            result = runner.invoke(artifact, ['get', 'incomplete/path'])
            assert result.exit_code != 0

    def test_cp_command_invalid_paths(self):
        """Test cp command with invalid path combinations."""
        runner = CliRunner()
        
        with patch('clarifai.cli.artifact.validate_context'):
            # Both paths are local
            result = runner.invoke(artifact, ['cp', './local1.txt', './local2.txt'])
            assert result.exit_code != 0
            assert "One of source or destination must be a local path" in result.output
            
            # Both paths are remote
            result = runner.invoke(artifact, [
                'cp', 
                'users/u1/apps/a1/artifacts/art1', 
                'users/u2/apps/a2/artifacts/art2'
            ])
            assert result.exit_code != 0
            assert "One of source or destination must be a local path" in result.output


if __name__ == "__main__":
    pytest.main([__file__])
