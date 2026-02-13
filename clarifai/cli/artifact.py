import os
import re
import shutil
from datetime import datetime
from typing import Dict, Optional

import click
from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from clarifai.cli.base import cli
from clarifai.client.artifact import Artifact
from clarifai.client.artifact_version import ArtifactVersion
from clarifai.constants.artifact import (
    ARTIFACT_VISIBILITY_ORG,
    ARTIFACT_VISIBILITY_PRIVATE,
    ARTIFACT_VISIBILITY_PUBLIC,
    RFC3339_FORMAT,
    RFC3339_FORMAT_NO_MICROSECONDS,
)
from clarifai.errors import UserError
from clarifai.utils.cli import AliasedGroup, display_co_resources, validate_context
from clarifai.utils.logging import logger

# Store builtin list to avoid name conflicts with command function
builtin_list = list

# Regex pattern for artifact paths covering all formats:
# - users/<user-id>/apps/<app-id>/artifacts/<artifact-id> (artifact-level)
# - users/<user-id>/apps/<app-id>/artifacts/<artifact-id>/versions/<version-id> (version-level)
ARTIFACT_PATH_PATTERN = (
    r'^users/([^/]+)/apps/([^/]+)(?:/artifacts/([^/]+)(?:/versions/([^/]+))?)?$'
)


def is_local_path(path: str) -> bool:
    """Check if a path refers to a local file/directory.

    Artifact paths follow the pattern:
    - users/<user-id>/apps/<app-id>/artifacts/<artifact-id>[/versions/<version-id>]
    Any path that doesn't match these patterns is considered a local path.
    """
    # Normalize path by removing leading/trailing slashes
    normalized_path = path.strip('/')

    # Check for artifact paths - they must contain both 'users/' and 'apps/'
    if normalized_path.startswith('users/') and '/apps/' in normalized_path:
        if re.match(ARTIFACT_PATH_PATTERN, normalized_path):
            return False
    return True


def parse_artifact_path(path: str) -> Dict[str, Optional[str]]:
    """Parse an artifact path in various formats:
    - users/<user-id>/apps/<app-id>/artifacts/<artifact-id> (artifact-level)
    - users/<user-id>/apps/<app-id>/artifacts/<artifact-id>/versions/<version-id> (version-level)

    Args:
        path: The artifact path to parse

    Returns:
        A dictionary with parsed components

    Raises:
        UserError: If the path format is invalid
    """
    # Normalize path by removing leading and trailing slashes
    normalized_path = path.strip('/')

    match = re.match(ARTIFACT_PATH_PATTERN, normalized_path)

    if match:
        user_id, app_id, artifact_id, version_id = match.groups()
        return {
            'user_id': user_id,
            'app_id': app_id,
            'artifact_id': artifact_id,
            'version_id': version_id,
        }

    raise UserError(
        "Invalid artifact path format. Expected: users/<user-id>/apps/<app-id>[/artifacts/<artifact-id>[/versions/<version-id>]]"
    )


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
    except ValueError:
        raise UserError(
            f"Invalid RFC3339 timestamp format: {timestamp_str}. Expected format: 2024-12-31T23:59:59.999Z or 2024-12-31T23:59:59Z"
        )


def _parse_and_validate_path(path, require_artifact=True, operation_type="general"):
    """Helper function to parse and validate artifact path."""
    parsed = parse_artifact_path(path)

    if require_artifact:
        # For all operations including upload, require artifact_id
        if not all([parsed['user_id'], parsed['app_id'], parsed['artifact_id']]):
            raise UserError("Path must include user_id, app_id, and artifact_id")

    return parsed


def _upload_artifact(source_path: str, parsed_destination: dict, client_kwargs: dict, **kwargs):
    """Upload a file to an artifact.

    Args:
        source_path: Local file path to upload
        parsed_destination: Pre-parsed destination path components
        client_kwargs: Client configuration (pat, base)
        **kwargs: Additional keyword arguments (description, visibility, expires_at)

    Returns:
        ArtifactVersion: The created artifact version
    """
    user_id = parsed_destination['user_id']
    app_id = parsed_destination['app_id']
    artifact_id = parsed_destination['artifact_id']  # Now required
    version_id = parsed_destination['version_id']  # May be None

    # Create artifact version and upload
    version = ArtifactVersion(
        artifact_id=artifact_id,
        version_id=version_id or "",
        user_id=user_id,
        app_id=app_id,
        **client_kwargs,
    )

    return version.upload(
        file_path=source_path,
        artifact_id=artifact_id,
        description=kwargs.get('description', ''),
        visibility=kwargs.get('visibility'),  # Pass None if not provided
        expires_at=kwargs.get('expires_at'),
        version_id=version_id,
    )


def _download_artifact(
    destination_path: str, parsed_source: dict, client_kwargs: dict, **kwargs
) -> str:
    """Download a file from an artifact.

    Args:
        destination_path: Local directory path to save to
        parsed_source: Pre-parsed source path components
        client_kwargs: Client configuration (pat, base)
        **kwargs: Additional keyword arguments (force)

    Returns:
        str: The path where file was downloaded
    """
    user_id = parsed_source['user_id']
    app_id = parsed_source['app_id']
    artifact_id = parsed_source['artifact_id']
    version_id = parsed_source['version_id']  # May be None for latest version

    if version_id:
        # Download specific version - use the provided version_id directly
        final_version_id = version_id
    else:
        # Download latest version - get artifact info to find latest version
        artifact = Artifact(
            artifact_id=artifact_id, user_id=user_id, app_id=app_id, **client_kwargs
        )
        artifact_info = artifact.get()
        if not artifact_info.artifact_version or not artifact_info.artifact_version.id:
            raise UserError("No versions available for this artifact")

        final_version_id = artifact_info.artifact_version.id

    # Create artifact version and download
    version = ArtifactVersion(
        artifact_id=artifact_id,
        version_id=final_version_id,
        user_id=user_id,
        app_id=app_id,
        **client_kwargs,
    )

    force = kwargs.get('force', False)
    return version.download(output_path=destination_path, force=force)


@cli.group(
    ['artifact', 'af'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def artifact():
    """Manage Artifacts: create, upload, download, list, get, delete"""


@artifact.command(['list', 'ls'])
@click.argument('path')
@click.option('--versions', is_flag=True, help='List artifact versions instead of artifacts')
@click.pass_context
def list(ctx, path, versions):
    """List artifacts or artifact versions.

    Examples:
        clarifai af list users/u/apps/a
        clarifai af list users/u/apps/a/artifacts/my-artifact --versions
    """
    try:
        validate_context(ctx)

        # Parse path and extract components
        if versions:
            parsed = parse_artifact_path(path)
            if not parsed['artifact_id']:
                click.echo(
                    "When using --versions, path must include artifact ID: users/<user-id>/apps/<app-id>/artifacts/<artifact-id>",
                    err=True,
                )
                raise click.Abort()

            # List artifact versions with full data
            artifact_version = ArtifactVersion(
                artifact_id=parsed['artifact_id'],
                user_id=parsed['user_id'],
                app_id=parsed['app_id'],
                pat=ctx.obj.current.pat,
                base=ctx.obj.current.api_base,
            )
            versions_list = builtin_list(artifact_version.list())

            if not versions_list:
                click.echo("No artifact versions found")
                return

            display_co_resources(
                versions_list,
                custom_columns={
                    'VERSION': lambda v: v.id,
                    'VISIBILITY': lambda v: resources_pb2.Visibility.Gettable.Name(
                        v.visibility.gettable
                    ),
                    'EXPIRES_AT': lambda v: 'Never'
                    if v.expires_at and v.expires_at.seconds == 0 and v.expires_at.nanos == 0
                    else str(v.expires_at.ToDatetime())
                    if v.expires_at
                    else 'Never',
                    'CREATED_AT': lambda v: str(v.created_at.ToDatetime()) if v.created_at else '',
                },
            )
        else:
            # For listing artifacts, we expect app-level path: users/<user-id>/apps/<app-id>
            if '/artifacts/' in path:
                click.echo("To list artifacts, use: users/<user-id>/apps/<app-id>", err=True)
                raise click.Abort()

            # Parse app-level path
            parsed = parse_artifact_path(path)

            # Validate it's an app-level path (no artifact_id)
            if parsed['artifact_id'] is not None:
                click.echo("To list artifacts, use: users/<user-id>/apps/<app-id>", err=True)
                raise click.Abort()

            # List artifacts
            artifact = Artifact(
                user_id=parsed['user_id'],
                app_id=parsed['app_id'],
                pat=ctx.obj.current.pat,
                base=ctx.obj.current.api_base,
            )
            artifacts_list = builtin_list(artifact.list())

            if not artifacts_list:
                click.echo("No artifacts found")
                return

            display_co_resources(
                artifacts_list,
                custom_columns={
                    'ARTIFACT': lambda a: a.id,
                    'LATEST_VERSION': lambda a: a.artifact_version.id
                    if a.artifact_version and a.artifact_version.id
                    else 'N/A',
                    'CREATED_AT': lambda a: str(a.created_at.ToDatetime()) if a.created_at else '',
                },
            )

    except UserError as e:
        click.echo(str(e), err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error listing artifacts: {e}", err=True)
        raise click.Abort()


@artifact.command()
@click.argument('path')
@click.pass_context
def get(ctx, path):
    """Get artifact or artifact version details.

    Examples:
        clarifai af get users/u/apps/a/artifacts/my-artifact
        clarifai af get users/u/apps/a/artifacts/my-artifact/versions/v123
    """
    try:
        validate_context(ctx)
        parsed = _parse_and_validate_path(path)
        version_id = parsed['version_id']

        if version_id:
            # Get artifact version
            artifact_version = ArtifactVersion(
                artifact_id=parsed['artifact_id'],
                version_id=version_id,
                user_id=parsed['user_id'],
                app_id=parsed['app_id'],
                pat=ctx.obj.current.pat,
                base=ctx.obj.current.api_base,
            )
            version_info = artifact_version.get()
            click.echo(f"Artifact Version: {version_id}")
            click.echo(
                f"Description: {version_info.description if version_info.description else 'N/A'}"
            )
            click.echo(
                f"Visibility: {resources_pb2.Visibility.Gettable.Name(version_info.visibility.gettable)}"
            )
            click.echo(
                f"Expires at: {'Never' if version_info.expires_at and version_info.expires_at.seconds == 0 and version_info.expires_at.nanos == 0 else version_info.expires_at.ToDatetime() if version_info.expires_at else 'Never'}"
            )
            click.echo(
                f"Created at: {version_info.created_at.ToDatetime() if version_info.created_at else 'N/A'}"
            )
            click.echo(
                f"Modified at: {version_info.modified_at.ToDatetime() if version_info.modified_at else 'N/A'}"
            )
        else:
            # Get artifact
            artifact = Artifact(
                artifact_id=parsed['artifact_id'],
                user_id=parsed['user_id'],
                app_id=parsed['app_id'],
                pat=ctx.obj.current.pat,
                base=ctx.obj.current.api_base,
            )
            artifact_info = artifact.get()
            click.echo(f"Artifact ID: {parsed['artifact_id']}")
            if artifact_info.artifact_version and artifact_info.artifact_version.id:
                click.echo(f"Latest version: {artifact_info.artifact_version.id}")
            click.echo(
                f"Created at: {artifact_info.created_at.ToDatetime() if artifact_info.created_at else 'N/A'}"
            )
            click.echo(
                f"Modified at: {artifact_info.modified_at.ToDatetime() if artifact_info.modified_at else 'N/A'}"
            )

    except UserError as e:
        click.echo(str(e), err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error getting artifact information: {e}", err=True)
        raise click.Abort()


@artifact.command(['delete', 'rm'])
@click.argument('path')
@click.option('-f', '--force', is_flag=True, help='Force deletion without confirmation')
@click.pass_context
def delete(ctx, path, force):
    """Delete an artifact or artifact version.

    Examples:
        clarifai af rm users/u/apps/a/artifacts/my-artifact
        clarifai af rm users/u/apps/a/artifacts/my-artifact/versions/v123
        clarifai af rm users/u/apps/a/artifacts/my-artifact --force
    """
    try:
        validate_context(ctx)
        parsed = _parse_and_validate_path(path)
        version_id = parsed['version_id']

        # Ask for confirmation unless force flag is used
        if not force:
            if version_id:
                prompt_msg = f"Are you sure you want to delete artifact version '{version_id}'?"
            else:
                prompt_msg = f"Are you sure you want to delete artifact '{parsed['artifact_id']}'?"
            if not click.confirm(prompt_msg):
                click.echo("Operation cancelled")
                return

        if version_id:
            # Delete artifact version
            artifact_version = ArtifactVersion(
                artifact_id=parsed['artifact_id'],
                version_id=version_id,
                user_id=parsed['user_id'],
                app_id=parsed['app_id'],
                pat=ctx.obj.current.pat,
                base=ctx.obj.current.api_base,
            )
            artifact_version.delete()
            click.echo(f"Successfully deleted artifact version {version_id}")
        else:
            # Delete artifact
            artifact = Artifact(
                artifact_id=parsed['artifact_id'],
                user_id=parsed['user_id'],
                app_id=parsed['app_id'],
                pat=ctx.obj.current.pat,
                base=ctx.obj.current.api_base,
            )
            artifact.delete()
            click.echo(f"Successfully deleted artifact {parsed['artifact_id']}")

    except UserError as e:
        click.echo(str(e), err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error deleting artifact: {e}", err=True)
        raise click.Abort()


@artifact.command()
@click.argument('source')
@click.argument('destination')
@click.option('--description', help='Description for the artifact version')
@click.option(
    '--visibility',
    type=click.Choice(
        [ARTIFACT_VISIBILITY_PRIVATE, ARTIFACT_VISIBILITY_PUBLIC, ARTIFACT_VISIBILITY_ORG]
    ),
    default=None,
    help='Visibility setting (defaults to "private")',
)
@click.option(
    '--expires-at', help='Expiration timestamp (RFC3339 format: 2024-12-31T23:59:59.999Z)'
)
@click.option('-f', '--force', is_flag=True, help='Force overwrite existing files')
@click.pass_context
def cp(
    ctx,
    source,
    destination,
    description,
    visibility,
    expires_at,
    force,
):
    """Upload or download artifact files.

    Upload examples:
        clarifai af cp ./model.pt users/u/apps/a/artifacts/my-artifact
        clarifai af cp /tmp/model.pt users/u/apps/a/artifacts/my-artifact --description="Version 2"
        clarifai af cp model.pt users/u/apps/a/artifacts/my-artifact/versions/v123

    Download examples:
        clarifai af cp users/u/apps/a/artifacts/my-artifact ./downloads/
        clarifai af cp users/u/apps/a/artifacts/my-artifact/versions/v123 /tmp/
        clarifai af cp users/u/apps/a/artifacts/my-artifact .
    """
    try:
        validate_context(ctx)

        # Validate input arguments
        if not source:
            raise UserError("source is required")
        if not destination:
            raise UserError("destination is required")

        # Determine operation type based on source and destination
        source_is_local = is_local_path(source)
        dest_is_local = is_local_path(destination)

        # Validate operation type
        if source_is_local and dest_is_local:
            raise UserError("One of source or destination must be an artifact path")
        elif not source_is_local and not dest_is_local:
            raise UserError("One of source or destination must be a local path")

        # Extract client params
        client_kwargs = {
            'pat': ctx.obj.current.pat,
            'base': ctx.obj.current.api_base,
        }

        if source_is_local and not dest_is_local:
            # Upload operation
            # Validate source file exists
            if not os.path.exists(source):
                raise UserError(f"Source file does not exist: {source}")

            # Parse and validate destination path
            parsed_destination = _parse_and_validate_path(
                destination, require_artifact=True, operation_type="upload"
            )

            # Parse expires_at if provided
            expires_at_timestamp = None
            if expires_at:
                try:
                    expires_at_timestamp = parse_rfc3339_timestamp(expires_at)
                    logger.info(f"Parsed expiration time: {expires_at}")
                except UserError as e:
                    click.echo(f"Error parsing expires_at: {e}", err=True)
                    raise click.Abort()

            # Call upload function with pre-validated data
            uploaded_version = _upload_artifact(
                source_path=source,
                parsed_destination=parsed_destination,
                client_kwargs=client_kwargs,
                description=description or "",
                visibility=visibility,
                expires_at=expires_at_timestamp,
            )

            click.echo(f"Successfully uploaded {source} to {destination}")
            if uploaded_version.version_id:
                click.echo(f"Version ID: {uploaded_version.version_id}")

        elif not source_is_local and dest_is_local:
            # Download operation
            # Parse and validate source path
            parsed_source = parse_artifact_path(source)
            if not all(
                [parsed_source['user_id'], parsed_source['app_id'], parsed_source['artifact_id']]
            ):
                raise UserError("user_id, app_id, and artifact_id are required for download")

            # Call download function with pre-validated data
            downloaded_path = _download_artifact(
                destination_path=destination,
                parsed_source=parsed_source,
                client_kwargs=client_kwargs,
                force=force,
            )

            click.echo(f"Successfully downloaded to {downloaded_path}")

    except UserError as e:
        click.echo(str(e), err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error processing file: {e}", err=True)
        raise click.Abort()
