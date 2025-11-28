"""Test file for artifact and artifact version functionality."""

import pytest
from unittest.mock import Mock, patch

from clarifai.client.artifact import Artifact, ArtifactVersion
from clarifai.errors import UserError


class TestArtifact:
    """Test class for Artifact client."""

    def test_init(self):
        """Test artifact initialization."""
        artifact = Artifact(
            artifact_id="test_artifact",
            user_id="test_user",
            app_id="test_app"
        )
        
        assert artifact.artifact_id == "test_artifact"
        assert artifact.user_id == "test_user"
        assert artifact.app_id == "test_app"
        assert artifact.id == "test_artifact"

    def test_repr(self):
        """Test artifact string representation."""
        artifact = Artifact(
            artifact_id="test_artifact",
            user_id="test_user",
            app_id="test_app"
        )
        
        repr_str = repr(artifact)
        assert "test_artifact" in repr_str
        assert "test_user" in repr_str
        assert "test_app" in repr_str


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

    def test_upload_missing_file(self):
        """Test upload with missing file."""
        with patch('clarifai.client.artifact.BaseClient.__init__', return_value=None):
            version = ArtifactVersion()
            version.auth_helper = Mock()
            
            with pytest.raises(UserError, match="File does not exist"):
                version.upload(
                    file_path="nonexistent_file.txt",
                    artifact_id="test_artifact",
                    user_id="test_user",
                    app_id="test_app"
                )

    def test_upload_missing_params(self):
        """Test upload with missing required parameters."""
        with patch('clarifai.client.artifact.BaseClient.__init__', return_value=None):
            version = ArtifactVersion()
            version.auth_helper = Mock()
            
            with pytest.raises(UserError, match="artifact_id is required"):
                version.upload(file_path="test.txt")


if __name__ == "__main__":
    pytest.main([__file__])
