import os
import re
from datetime import datetime
from typing import Dict, Optional

from google.protobuf.timestamp_pb2 import Timestamp

from clarifai.client.artifact import Artifact
from clarifai.client.artifact_version import ArtifactVersion
from clarifai.client.base import BaseClient
from clarifai.constants.artifact import (
    RFC3339_FORMAT,
    RFC3339_FORMAT_NO_MICROSECONDS,
)
from clarifai.errors import UserError
from clarifai.utils.logging import logger


def is_local_path(path: str) -> bool:
    """Check if a path refers to a local file/directory."""
    # Check for URL schemes
    if path.startswith(('http://', 'https://', 'ftp://', 'ftps://')):
        return False
    # Check for artifact paths
    if path.startswith('users/'):
        return False
    return True


def parse_artifact_path(path: str) -> Dict[str, Optional[str]]:
    """Parse an artifact path like users/<user-id>/apps/<app-id>/artifacts/<artifact-id>[/versions/<version-id>]

    Args:
        path: The artifact path to parse

    Returns:
        A dictionary with parsed components

    Raises:
        UserError: If the path format is invalid
    """
    # Match users/u/apps/a/artifacts/art_id or users/u/apps/a/artifacts/art_id/versions/ver_id
    pattern = r'^users/([^/]+)/apps/([^/]+)/artifacts/([^/]+)(?:/versions/([^/]+))?/?$'
    match = re.match(pattern, path.rstrip('/'))

    if not match:
        raise UserError(
            "Invalid artifact path format. Expected: users/<user-id>/apps/<app-id>/artifacts/<artifact-id>[/versions/<version-id>]"
        )

    user_id, app_id, artifact_id, version_id = match.groups()

    return {
        'user_id': user_id,
        'app_id': app_id,
        'artifact_id': artifact_id,
        'version_id': version_id,
    }


def parse_rfc3339_timestamp(timestamp_str: str) -> Timestamp:
    """Parse RFC3339 timestamp string to protobuf Timestamp.

    Args:
        timestamp_str: RFC3339 formatted timestamp (e.g., "2024-12-31T23:59:59.999Z")

    Returns:
        Protobuf Timestamp object

    Raises:
        UserError: If the timestamp format is invalid
    """
    if not timestamp_str:
        return None

    try:
        # Try with microseconds first
        try:
            dt = datetime.strptime(timestamp_str, RFC3339_FORMAT)
        except ValueError:
            # Try without microseconds
            dt = datetime.strptime(timestamp_str, RFC3339_FORMAT_NO_MICROSECONDS)

        timestamp = Timestamp()
        timestamp.FromDatetime(dt)
        return timestamp
    except ValueError as e:
        raise UserError(
            f"Invalid RFC3339 timestamp format: {timestamp_str}. Expected format: 2024-12-31T23:59:59.999Z or 2024-12-31T23:59:59Z"
        )


class ArtifactBuilder(BaseClient):
    """Artifact Builder for complex artifact workflows like upload/download with path parsing."""

    def __init__(self, **kwargs):
        """Initialize the ArtifactBuilder.

        Args:
            **kwargs: Additional keyword arguments to be passed to the BaseClient.
        """
        super().__init__(**kwargs)

    @property
    def pat(self) -> str:
        """Get the personal access token from auth_helper."""
        return self.auth_helper.pat

    @pat.setter
    def pat(self, value: str):
        """Set the personal access token."""
        self.auth_helper.pat = value

    def upload_from_path(
        self,
        source_path: str,
        destination_path: str,
        description: str = "",
        visibility: str = "private",
        expires_at: Optional[Timestamp] = None,
        version_id: str = "",
        user_id: str = "",
        app_id: str = "",
    ) -> ArtifactVersion:
        """Upload a file to an artifact from local path to destination path.

        This is a convenience method that handles path parsing and artifact creation.

        Args:
            source_path: Local file path to upload
            destination_path: Artifact path in format users/<user-id>/apps/<app-id>/artifacts/<artifact-id>
            description: Description for the artifact version
            visibility: Visibility setting ('private' or 'public')
            expires_at: Expiration timestamp
            version_id: Version ID to assign (optional)
            user_id: User ID (optional, extracted from path if not provided)
            app_id: App ID (optional, extracted from path if not provided)

        Returns:
            ArtifactVersion: The created artifact version

        Raises:
            UserError: If source file doesn't exist or path is invalid
        """
        if not source_path:
            raise UserError("source_path is required")
        if not destination_path:
            raise UserError("destination_path is required")

        # Validate paths
        if not is_local_path(source_path):
            raise UserError("source_path must be a local path")
        if is_local_path(destination_path):
            raise UserError("destination_path must be an artifact path")

        if not os.path.exists(source_path):
            raise UserError(f"Source file does not exist: {source_path}")

        # Parse destination path
        parsed = parse_artifact_path(destination_path)
        user_id = user_id or parsed['user_id']
        app_id = app_id or parsed['app_id']
        artifact_id = parsed['artifact_id']

        if not all([user_id, app_id, artifact_id]):
            raise UserError("user_id, app_id, and artifact_id are required for upload")

        # Check if artifact exists, create if it doesn't
        artifact = Artifact(
            artifact_id=artifact_id, user_id=user_id, app_id=app_id, **self._get_client_params()
        )

        if not artifact.exists():
            logger.info(f"Creating artifact {artifact_id}")
            artifact.create(artifact_id=artifact_id, user_id=user_id, app_id=app_id)

        # Create new artifact version and upload
        version = ArtifactVersion(
            artifact_id=artifact_id, user_id=user_id, app_id=app_id, **self._get_client_params()
        )

        return version.create(
            file_path=source_path,
            artifact_id=artifact_id,
            description=description,
            visibility=visibility,
            expires_at=expires_at,
            version_id=version_id,
            user_id=user_id,
            app_id=app_id,
        )

    def download_from_path(
        self,
        source_path: str,
        destination_path: str,
        user_id: str = "",
        app_id: str = "",
        force: bool = False,
    ) -> str:
        """Download a file from artifact path to local destination.

        This is a convenience method that handles path parsing.

        Args:
            source_path: Artifact path in format users/<user-id>/apps/<app-id>/artifacts/<artifact-id>[/versions/<version-id>]
            destination_path: Local file path to save to
            user_id: User ID (optional, extracted from path if not provided)
            app_id: App ID (optional, extracted from path if not provided)
            force: Force overwrite existing files

        Returns:
            str: The path where file was downloaded

        Raises:
            UserError: If path is invalid or artifact not found
        """
        if not source_path:
            raise UserError("source_path is required")
        if not destination_path:
            raise UserError("destination_path is required")

        # Validate paths
        if is_local_path(source_path):
            raise UserError("source_path must be an artifact path")
        if not is_local_path(destination_path):
            raise UserError("destination_path must be a local path")

        # Parse source path
        parsed = parse_artifact_path(source_path)
        user_id = user_id or parsed['user_id']
        app_id = app_id or parsed['app_id']
        artifact_id = parsed['artifact_id']
        version_id = parsed['version_id']  # May be None for latest version

        if not all([user_id, app_id, artifact_id]):
            raise UserError("user_id, app_id, and artifact_id are required for download")

        if version_id:
            # Download specific version
            version = ArtifactVersion(
                artifact_id=artifact_id,
                version_id=version_id,
                user_id=user_id,
                app_id=app_id,
                **self._get_client_params(),
            )
        else:
            # Download latest version - get artifact info to find latest version
            artifact = Artifact(
                artifact_id=artifact_id,
                user_id=user_id,
                app_id=app_id,
                **self._get_client_params(),
            )
            info = artifact.info()
            if not info.get('artifact_version'):
                raise UserError("No versions available for this artifact")

            latest_version_id = info['artifact_version']['id']
            version = ArtifactVersion(
                artifact_id=artifact_id,
                version_id=latest_version_id,
                user_id=user_id,
                app_id=app_id,
                **self._get_client_params(),
            )

        # Download the file
        return version.download(output_path=destination_path, force=force)

    def list_artifacts(self, user_id: str, app_id: str, **kwargs):
        """List artifacts in an app.

        Args:
            user_id: The user ID
            app_id: The app ID
            **kwargs: Additional arguments

        Returns:
            Generator of Artifact objects
        """
        return Artifact.list(user_id=user_id, app_id=app_id, **kwargs)

    def list_artifact_versions(self, artifact_id: str, user_id: str, app_id: str, **kwargs):
        """List artifact versions.

        Args:
            artifact_id: The artifact ID
            user_id: The user ID
            app_id: The app ID
            **kwargs: Additional arguments

        Returns:
            Generator of ArtifactVersion objects
        """
        return ArtifactVersion.list(
            artifact_id=artifact_id, user_id=user_id, app_id=app_id, **kwargs
        )

    def get_artifact_info(self, artifact_id: str, user_id: str, app_id: str, **kwargs):
        """Get artifact information.

        Args:
            artifact_id: The artifact ID
            user_id: The user ID
            app_id: The app ID
            **kwargs: Additional arguments

        Returns:
            Artifact info dictionary
        """
        artifact = Artifact(artifact_id=artifact_id, user_id=user_id, app_id=app_id, **kwargs)
        return artifact.info()

    def get_artifact_version_info(
        self, artifact_id: str, version_id: str, user_id: str, app_id: str, **kwargs
    ):
        """Get artifact version information.

        Args:
            artifact_id: The artifact ID
            version_id: The version ID
            user_id: The user ID
            app_id: The app ID
            **kwargs: Additional arguments

        Returns:
            Artifact version info dictionary
        """
        version = ArtifactVersion(
            artifact_id=artifact_id,
            version_id=version_id,
            user_id=user_id,
            app_id=app_id,
            **kwargs,
        )
        return version.info()

    def delete_artifact(self, artifact_id: str, user_id: str, app_id: str, **kwargs):
        """Delete an artifact.

        Args:
            artifact_id: The artifact ID
            user_id: The user ID
            app_id: The app ID
            **kwargs: Additional arguments

        Returns:
            Boolean indicating success
        """
        artifact = Artifact(artifact_id=artifact_id, user_id=user_id, app_id=app_id, **kwargs)
        return artifact.delete()

    def delete_artifact_version(
        self, artifact_id: str, version_id: str, user_id: str, app_id: str, **kwargs
    ):
        """Delete an artifact version.

        Args:
            artifact_id: The artifact ID
            version_id: The version ID
            user_id: The user ID
            app_id: The app ID
            **kwargs: Additional arguments

        Returns:
            Boolean indicating success
        """
        version = ArtifactVersion(
            artifact_id=artifact_id,
            version_id=version_id,
            user_id=user_id,
            app_id=app_id,
            **kwargs,
        )
        return version.delete()

    def _get_client_params(self) -> Dict:
        """Get the client parameters for creating new instances."""
        return {
            "pat": self.pat,
            "base": self.base,
        }


def upload_artifact(source_path: str, destination_path: str, **kwargs) -> ArtifactVersion:
    """Convenience function for uploading artifacts.

    Args:
        source_path: Local file path to upload
        destination_path: Artifact path
        **kwargs: Additional keyword arguments

    Returns:
        ArtifactVersion: The uploaded artifact version
    """
    # Filter kwargs for builder constructor - only pass authentication and config args
    method_kwargs = [
        'description',
        'visibility',
        'expires_at',
        'version_id',
        'user_id',
        'app_id',
        'force',
    ]
    builder_kwargs = {k: v for k, v in kwargs.items() if k not in method_kwargs}
    upload_kwargs = {k: v for k, v in kwargs.items() if k in method_kwargs}
    builder = ArtifactBuilder(**builder_kwargs)
    return builder.upload_from_path(source_path, destination_path, **upload_kwargs)


def download_artifact(source_path: str, destination_path: str, **kwargs) -> str:
    """Convenience function for downloading artifacts.

    Args:
        source_path: Artifact path
        destination_path: Local file path to save to
        **kwargs: Additional keyword arguments

    Returns:
        str: The path where file was downloaded
    """
    # Filter kwargs for builder constructor - only pass authentication and config args
    method_kwargs = ['force', 'user_id', 'app_id']
    builder_kwargs = {k: v for k, v in kwargs.items() if k not in method_kwargs}
    download_kwargs = {k: v for k, v in kwargs.items() if k in method_kwargs}
    builder = ArtifactBuilder(**builder_kwargs)
    return builder.download_from_path(source_path, destination_path, **download_kwargs)
