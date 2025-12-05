import shutil

import click

from clarifai.cli.base import cli
from clarifai.client.artifact import Artifact
from clarifai.client.artifact_version import ArtifactVersion
from clarifai.errors import UserError
from clarifai.runners.artifacts.artifact_builder import (
    parse_artifact_path,
    parse_rfc3339_timestamp,
)
from clarifai.utils.cli import AliasedGroup, TableFormatter, validate_context
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


@cli.group(
    ['artifact', 'af'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def artifact():
    """Manage Artifacts: create, upload, download, list, get, delete"""


@artifact.command(['list', 'ls'])
@click.argument('path', required=False)
@click.option('--user-id', help='User ID')
@click.option('--app-id', help='App ID')
@click.option('--artifact-id', help='Artifact ID')
@click.option('--versions', is_flag=True, help='List artifact versions instead of artifacts')
@click.pass_context
def list(ctx, path, user_id, app_id, artifact_id, versions):
    """List artifacts or artifact versions.

    Examples:
        clarifai artifact list users/u/apps/a
        clarifai artifact list --app-id=a --user-id=u
        clarifai artifact list users/u/apps/a/artifacts/my-artifact --versions
    """
    try:
        validate_context(ctx)

        if path:
            parsed = parse_artifact_path(
                path + ('/artifacts/dummy' if '/artifacts/' not in path else '')
            )
            user_id = user_id or parsed['user_id']
            app_id = app_id or parsed['app_id']
            if '/artifacts/' in path and versions:
                artifact_id = artifact_id or parsed['artifact_id']

        if not user_id or not app_id:
            click.echo("user_id and app_id are required", err=True)
            raise click.Abort()

        if versions:
            if not artifact_id:
                click.echo("artifact_id is required when listing versions", err=True)
                raise click.Abort()

            # Use ArtifactVersion client to list versions
            versions_list = list(
                ArtifactVersion.list(
                    artifact_id=artifact_id,
                    user_id=user_id,
                    app_id=app_id,
                    **ctx.obj.current.to_grpc(),
                )
            )

            # Display versions in a table
            if not versions_list:
                click.echo("No artifact versions found")
                return

            table_data = []
            for version in versions_list:
                info = version.info()
                table_data.append(
                    {
                        'VERSION_ID': version.version_id,
                        'DESCRIPTION': info.get('description', ''),
                        'VISIBILITY': info.get('visibility', 'UNKNOWN'),
                        'CREATED_AT': str(info.get('created_at', '')),
                    }
                )

            if table_data:
                from collections import OrderedDict

                columns = OrderedDict(
                    [
                        ('VERSION_ID', lambda x: x['VERSION_ID']),
                        ('DESCRIPTION', lambda x: x['DESCRIPTION']),
                        ('VISIBILITY', lambda x: x['VISIBILITY']),
                        ('CREATED_AT', lambda x: x['CREATED_AT']),
                    ]
                )
                formatter = TableFormatter(custom_columns=columns)
                print(formatter.format(table_data))
        else:
            # Use Artifact client to list artifacts
            artifacts_list = list(
                Artifact.list(user_id=user_id, app_id=app_id, **ctx.obj.current.to_grpc())
            )

            if not artifacts_list:
                click.echo("No artifacts found")
                return

            # Display artifacts in a table
            if not artifacts_list:
                click.echo("No artifacts found")
                return

            table_data = []
            for artifact_obj in artifacts_list:
                info = artifact_obj.info()
                table_data.append(
                    {
                        'ARTIFACT_ID': artifact_obj.artifact_id,
                        'USER_ID': info.get('user_id', ''),
                        'APP_ID': info.get('app_id', ''),
                        'CREATED_AT': str(info.get('created_at', '')),
                    }
                )

            if table_data:
                from collections import OrderedDict

                columns = OrderedDict(
                    [
                        ('ARTIFACT_ID', lambda x: x['ARTIFACT_ID']),
                        ('USER_ID', lambda x: x['USER_ID']),
                        ('APP_ID', lambda x: x['APP_ID']),
                        ('CREATED_AT', lambda x: x['CREATED_AT']),
                    ]
                )
                formatter = TableFormatter(custom_columns=columns)
                print(formatter.format(table_data))

    except UserError as e:
        click.echo(str(e), err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error listing artifacts: {e}", err=True)
        raise click.Abort()


@artifact.command()
@click.argument('path')
@click.option('--user-id', help='User ID')
@click.option('--app-id', help='App ID')
@click.option('--artifact-id', help='Artifact ID')
@click.option('--version-id', help='Artifact version ID')
@click.pass_context
def get(ctx, path, user_id, app_id, artifact_id, version_id):
    """Get artifact or artifact version details.

    Examples:
        clarifai artifact get users/u/apps/a/artifacts/my-artifact
        clarifai artifact get users/u/apps/a/artifacts/my-artifact/versions/v123
        clarifai artifact get --app-id=a --user-id=u --artifact-id=my-artifact
    """
    try:
        validate_context(ctx)

        if path:
            parsed = parse_artifact_path(path)
            user_id = user_id or parsed['user_id']
            app_id = app_id or parsed['app_id']
            artifact_id = artifact_id or parsed['artifact_id']
            if parsed['version_id']:
                version_id = version_id or parsed['version_id']

        if not all([user_id, app_id, artifact_id]):
            click.echo("user_id, app_id, and artifact_id are required", err=True)
            raise click.Abort()
        if version_id:
            # Get artifact version using ArtifactVersion client
            version = ArtifactVersion(
                artifact_id=artifact_id,
                version_id=version_id,
                user_id=user_id,
                app_id=app_id,
                **ctx.obj.current.to_grpc(),
            )
            info = version.info()
            click.echo(f"Artifact Version: {version_id}")
            click.echo(f"Description: {info.get('description', 'N/A')}")
            click.echo(f"Visibility: {info.get('visibility', 'UNKNOWN')}")
            click.echo(f"Created at: {info.get('created_at', 'N/A')}")
            click.echo(f"Modified at: {info.get('modified_at', 'N/A')}")

            if info.get('upload'):
                upload = info['upload']
                click.echo(f"Upload ID: {upload.get('id', 'N/A')}")
                click.echo(f"File name: {upload.get('content_name', 'N/A')}")
                click.echo(f"File size: {upload.get('content_length', 'N/A')} bytes")
                click.echo(f"Upload status: {upload.get('status', 'UNKNOWN')}")
        else:
            # Get artifact using Artifact client
            artifact = Artifact(
                artifact_id=artifact_id,
                user_id=user_id,
                app_id=app_id,
                **ctx.obj.current.to_grpc(),
            )
            info = artifact.info()
            click.echo(f"Artifact ID: {artifact_id}")
            click.echo(f"User ID: {info.get('user_id', 'N/A')}")
            click.echo(f"App ID: {info.get('app_id', 'N/A')}")
            click.echo(f"Created at: {info.get('created_at', 'N/A')}")
            click.echo(f"Modified at: {info.get('modified_at', 'N/A')}")

            if info.get('artifact_version'):
                version_info = info['artifact_version']
                click.echo(f"Latest version: {version_info.get('id', 'N/A')}")
                click.echo(f"Latest description: {version_info.get('description', 'N/A')}")

    except UserError as e:
        click.echo(str(e), err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error getting artifact information: {e}", err=True)
        raise click.Abort()


@artifact.command()
@click.argument('path')
@click.option('--user-id', help='User ID')
@click.option('--app-id', help='App ID')
@click.option('--artifact-id', help='Artifact ID')
@click.option('--version-id', help='Artifact version ID')
@click.pass_context
def delete(ctx, path, user_id, app_id, artifact_id, version_id):
    """Delete an artifact or artifact version.

    Examples:
        clarifai artifact delete users/u/apps/a/artifacts/my-artifact
        clarifai artifact delete users/u/apps/a/artifacts/my-artifact/versions/v123
    """
    try:
        validate_context(ctx)

        if path:
            parsed = parse_artifact_path(path)
            user_id = user_id or parsed['user_id']
            app_id = app_id or parsed['app_id']
            artifact_id = artifact_id or parsed['artifact_id']
            if parsed['version_id']:
                version_id = version_id or parsed['version_id']

        if not all([user_id, app_id, artifact_id]):
            click.echo("user_id, app_id, and artifact_id are required", err=True)
            raise click.Abort()

        # Ask for confirmation
        if not click.confirm('Are you sure you want to delete this artifact?'):
            click.echo("Operation cancelled")
            return

        if version_id:
            # Delete artifact version
            version = ArtifactVersion(
                artifact_id=artifact_id,
                version_id=version_id,
                user_id=user_id,
                app_id=app_id,
                **ctx.obj.current.to_grpc(),
            )
            version.delete()
            click.echo(f"Successfully deleted artifact version {version_id}")
        else:
            # Delete artifact
            artifact = Artifact(
                artifact_id=artifact_id,
                user_id=user_id,
                app_id=app_id,
                **ctx.obj.current.to_grpc(),
            )
            artifact.delete()
            click.echo(f"Successfully deleted artifact {artifact_id}")

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
    type=click.Choice(['private', 'public']),
    default='private',
    help='Visibility setting',
)
@click.option(
    '--expires-at', help='Expiration timestamp (RFC3339 format: 2024-12-31T23:59:59.999Z)'
)
@click.option('--version-id', help='Version ID to assign')
@click.option('--user-id', help='User ID')
@click.option('--app-id', help='App ID')
@click.option('-f', '--force', is_flag=True, help='Force overwrite existing files')
@click.pass_context
def cp(
    ctx,
    source,
    destination,
    description,
    visibility,
    expires_at,
    version_id,
    user_id,
    app_id,
    force,
):
    """Upload or download artifact files (S3-style cp command).

    Upload examples:
        clarifai artifact cp ./myfile.pt users/u/apps/a/artifacts/my-artifact
        clarifai artifact cp ./myfile.pt users/u/apps/a/artifacts/my-artifact --description="Version 2"

    Download examples:
        clarifai artifact cp users/u/apps/a/artifacts/my-artifact ./myfile.pt
        clarifai artifact cp users/u/apps/a/artifacts/my-artifact/versions/v123 ./myfile.pt
    """
    try:
        validate_context(ctx)

        # Determine operation type based on source and destination
        source_is_local = is_local_path(source)
        dest_is_local = is_local_path(destination)

        if source_is_local and not dest_is_local:
            # Upload operation
            # Parse expires_at if provided
            expires_at_timestamp = None
            if expires_at:
                try:
                    expires_at_timestamp = parse_rfc3339_timestamp(expires_at)
                    logger.info(f"Parsed expiration time: {expires_at}")
                except UserError as e:
                    click.echo(f"Error parsing expires_at: {e}", err=True)
                    raise click.Abort()

            # Use artifact builder for complex upload workflow
            from clarifai.runners.artifacts.artifact_builder import upload_artifact

            uploaded_version = upload_artifact(
                source_path=source,
                destination_path=destination,
                description=description or "",
                visibility=visibility,
                expires_at=expires_at_timestamp,
                version_id=version_id,
                user_id=user_id,
                app_id=app_id,
                **ctx.obj.current.to_grpc(),
            )

            click.echo(f"Successfully uploaded {source} to {destination}")
            if uploaded_version.version_id:
                click.echo(f"Version ID: {uploaded_version.version_id}")

        elif not source_is_local and dest_is_local:
            # Download operation
            # Use artifact builder for complex download workflow
            from clarifai.runners.artifacts.artifact_builder import download_artifact

            downloaded_path = download_artifact(
                source_path=source,
                destination_path=destination,
                user_id=user_id,
                app_id=app_id,
                force=force,
                **ctx.obj.current.to_grpc(),
            )

            click.echo(f"Successfully downloaded to {downloaded_path}")
        else:
            click.echo(
                "One of source or destination must be a local path and the other an artifact path",
                err=True,
            )
            raise click.Abort()

    except UserError as e:
        click.echo(str(e), err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error uploading file: {e}", err=True)
        raise click.Abort()
