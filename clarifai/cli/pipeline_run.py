import os
import shutil

import click

from clarifai.cli.base import cli
from clarifai.utils.cli import AliasedGroup, from_yaml, validate_context
from clarifai.utils.logging import logger


def _load_pipeline_params_from_config(user_id, app_id, pipeline_id, pipeline_version_id):
    """Load pipeline parameters from config-lock.yaml if not all provided.

    Args:
        user_id: User ID (may be None)
        app_id: App ID (may be None)
        pipeline_id: Pipeline ID (may be None)
        pipeline_version_id: Pipeline Version ID (may be None)

    Returns:
        tuple: (user_id, app_id, pipeline_id, pipeline_version_id)
    """
    if not all([user_id, app_id, pipeline_id, pipeline_version_id]):
        lockfile_path = os.path.join(os.getcwd(), "config-lock.yaml")
        if os.path.exists(lockfile_path):
            logger.info("Loading parameters from config-lock.yaml")
            lockfile_data = from_yaml(lockfile_path)

            if 'pipeline' in lockfile_data:
                pipeline_config = lockfile_data['pipeline']
                user_id = user_id or pipeline_config.get('user_id')
                app_id = app_id or pipeline_config.get('app_id')
                pipeline_id = pipeline_id or pipeline_config.get('id')
                pipeline_version_id = pipeline_version_id or pipeline_config.get('version_id')

    return user_id, app_id, pipeline_id, pipeline_version_id


def _validate_pipeline_params(user_id, app_id, pipeline_id, pipeline_version_id):
    """Validate that all required pipeline parameters are present.

    Args:
        user_id: User ID
        app_id: App ID
        pipeline_id: Pipeline ID
        pipeline_version_id: Pipeline Version ID

    Raises:
        click.UsageError: If any required parameter is missing
    """
    if not all([user_id, app_id, pipeline_id, pipeline_version_id]):
        raise click.UsageError(
            "Missing required parameters. Either provide --user_id, --app_id, "
            "--pipeline_id, and --pipeline_version_id, or ensure config-lock.yaml exists."
        )


def _create_pipeline(ctx, user_id, app_id, pipeline_id, pipeline_version_id):
    """Create and return a Pipeline object.

    Args:
        ctx: Click context
        user_id: User ID
        app_id: App ID
        pipeline_id: Pipeline ID
        pipeline_version_id: Pipeline Version ID

    Returns:
        Pipeline: Configured Pipeline object
    """
    from clarifai.client.pipeline import Pipeline

    return Pipeline(
        pipeline_id=pipeline_id,
        pipeline_version_id=pipeline_version_id,
        user_id=user_id,
        app_id=app_id,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
    )


@cli.group(
    ['pipelinerun', 'pr'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def pipelinerun():
    """Manage Pipeline Version Runs: pause, cancel, resume, monitor"""


@pipelinerun.command()
@click.argument('pipeline_version_run_id', required=False)
@click.option(
    '--pipeline_version_run_id',
    'pipeline_version_run_id_flag',
    required=False,
    help='Pipeline Version Run ID to pause.',
)
@click.option('--user_id', required=False, help='User ID that owns the pipeline.')
@click.option('--app_id', required=False, help='App ID that contains the pipeline.')
@click.option('--pipeline_id', required=False, help='Pipeline ID.')
@click.option('--pipeline_version_id', required=False, help='Pipeline Version ID.')
@click.pass_context
def pause(
    ctx,
    pipeline_version_run_id,
    pipeline_version_run_id_flag,
    user_id,
    app_id,
    pipeline_id,
    pipeline_version_id,
):
    """Pause a pipeline version run.

    Pausing is allowed only when the pipeline run is in Queued or Running state.

    Examples:

        # Using positional argument
        clarifai pr pause <pipeline_version_run_id>

        # Using flag
        clarifai pipelinerun pause --pipeline_version_run_id=<id>

        # With explicit parameters
        clarifai pr pause <pipeline_version_run_id> \\
            --user_id=USER_ID \\
            --app_id=APP_ID \\
            --pipeline_id=PIPELINE_ID \\
            --pipeline_version_id=VERSION_ID
    """
    from clarifai_grpc.grpc.api.status import status_code_pb2

    validate_context(ctx)

    # Resolve pipeline_version_run_id from positional or flag
    run_id = pipeline_version_run_id or pipeline_version_run_id_flag
    if not run_id:
        raise click.UsageError(
            "pipeline_version_run_id is required. "
            "Provide it as a positional argument or use --pipeline_version_run_id flag."
        )

    # Load parameters from config-lock.yaml if not provided
    user_id, app_id, pipeline_id, pipeline_version_id = _load_pipeline_params_from_config(
        user_id, app_id, pipeline_id, pipeline_version_id
    )

    # Validate required parameters
    _validate_pipeline_params(user_id, app_id, pipeline_id, pipeline_version_id)

    # Create Pipeline object
    pipeline = _create_pipeline(ctx, user_id, app_id, pipeline_id, pipeline_version_id)

    # Patch the pipeline version run to JOB_PAUSED
    try:
        result = pipeline.patch_pipeline_version_run(
            pipeline_version_run_id=run_id,
            orchestration_status_code=status_code_pb2.JOB_PAUSED,
        )
        logger.info(f"Successfully paused pipeline version run {run_id}")
        click.echo(f"Pipeline version run {run_id} has been paused.")
    except Exception as e:
        logger.error(f"Failed to pause pipeline version run: {e}")
        raise click.ClickException(str(e))


@pipelinerun.command()
@click.argument('pipeline_version_run_id', required=False)
@click.option(
    '--pipeline_version_run_id',
    'pipeline_version_run_id_flag',
    required=False,
    help='Pipeline Version Run ID to cancel.',
)
@click.option('--user_id', required=False, help='User ID that owns the pipeline.')
@click.option('--app_id', required=False, help='App ID that contains the pipeline.')
@click.option('--pipeline_id', required=False, help='Pipeline ID.')
@click.option('--pipeline_version_id', required=False, help='Pipeline Version ID.')
@click.pass_context
def cancel(
    ctx,
    pipeline_version_run_id,
    pipeline_version_run_id_flag,
    user_id,
    app_id,
    pipeline_id,
    pipeline_version_id,
):
    """Cancel a pipeline version run.

    Cancelling is allowed when the pipeline run is not already in a terminal state.

    Examples:

        # Using positional argument
        clarifai pr cancel <pipeline_version_run_id>

        # Using flag
        clarifai pipelinerun cancel --pipeline_version_run_id=<id>

        # With explicit parameters
        clarifai pr cancel <pipeline_version_run_id> \\
            --user_id=USER_ID \\
            --app_id=APP_ID \\
            --pipeline_id=PIPELINE_ID \\
            --pipeline_version_id=VERSION_ID
    """
    from clarifai_grpc.grpc.api.status import status_code_pb2

    validate_context(ctx)

    # Resolve pipeline_version_run_id from positional or flag
    run_id = pipeline_version_run_id or pipeline_version_run_id_flag
    if not run_id:
        raise click.UsageError(
            "pipeline_version_run_id is required. "
            "Provide it as a positional argument or use --pipeline_version_run_id flag."
        )

    # Load parameters from config-lock.yaml if not provided
    user_id, app_id, pipeline_id, pipeline_version_id = _load_pipeline_params_from_config(
        user_id, app_id, pipeline_id, pipeline_version_id
    )

    # Validate required parameters
    _validate_pipeline_params(user_id, app_id, pipeline_id, pipeline_version_id)

    # Create Pipeline object
    pipeline = _create_pipeline(ctx, user_id, app_id, pipeline_id, pipeline_version_id)

    # Patch the pipeline version run to JOB_CANCELLED
    try:
        result = pipeline.patch_pipeline_version_run(
            pipeline_version_run_id=run_id,
            orchestration_status_code=status_code_pb2.JOB_CANCELLED,
        )
        logger.info(f"Successfully cancelled pipeline version run {run_id}")
        click.echo(f"Pipeline version run {run_id} has been cancelled.")
    except Exception as e:
        logger.error(f"Failed to cancel pipeline version run: {e}")
        raise click.ClickException(str(e))


@pipelinerun.command()
@click.argument('pipeline_version_run_id', required=False)
@click.option(
    '--pipeline_version_run_id',
    'pipeline_version_run_id_flag',
    required=False,
    help='Pipeline Version Run ID to resume.',
)
@click.option('--user_id', required=False, help='User ID that owns the pipeline.')
@click.option('--app_id', required=False, help='App ID that contains the pipeline.')
@click.option('--pipeline_id', required=False, help='Pipeline ID.')
@click.option('--pipeline_version_id', required=False, help='Pipeline Version ID.')
@click.pass_context
def resume(
    ctx,
    pipeline_version_run_id,
    pipeline_version_run_id_flag,
    user_id,
    app_id,
    pipeline_id,
    pipeline_version_id,
):
    """Resume a paused pipeline version run.

    Resuming is allowed only when the pipeline run is in Paused state.

    Examples:

        # Using positional argument
        clarifai pr resume <pipeline_version_run_id>

        # Using flag
        clarifai pipelinerun resume --pipeline_version_run_id=<id>

        # With explicit parameters
        clarifai pr resume <pipeline_version_run_id> \\
            --user_id=USER_ID \\
            --app_id=APP_ID \\
            --pipeline_id=PIPELINE_ID \\
            --pipeline_version_id=VERSION_ID
    """
    from clarifai_grpc.grpc.api.status import status_code_pb2

    validate_context(ctx)

    # Resolve pipeline_version_run_id from positional or flag
    run_id = pipeline_version_run_id or pipeline_version_run_id_flag
    if not run_id:
        raise click.UsageError(
            "pipeline_version_run_id is required. "
            "Provide it as a positional argument or use --pipeline_version_run_id flag."
        )

    # Load parameters from config-lock.yaml if not provided
    user_id, app_id, pipeline_id, pipeline_version_id = _load_pipeline_params_from_config(
        user_id, app_id, pipeline_id, pipeline_version_id
    )

    # Validate required parameters
    _validate_pipeline_params(user_id, app_id, pipeline_id, pipeline_version_id)

    # Create Pipeline object
    pipeline = _create_pipeline(ctx, user_id, app_id, pipeline_id, pipeline_version_id)

    # Patch the pipeline version run to JOB_RUNNING to resume
    try:
        result = pipeline.patch_pipeline_version_run(
            pipeline_version_run_id=run_id,
            orchestration_status_code=status_code_pb2.JOB_RUNNING,
        )
        logger.info(f"Successfully resumed pipeline version run {run_id}")
        click.echo(f"Pipeline version run {run_id} has been resumed.")
    except Exception as e:
        logger.error(f"Failed to resume pipeline version run: {e}")
        raise click.ClickException(str(e))


@pipelinerun.command()
@click.argument('pipeline_version_run_id', required=False)
@click.option(
    '--pipeline_version_run_id',
    'pipeline_version_run_id_flag',
    required=False,
    help='Pipeline Version Run ID to monitor.',
)
@click.option('--user_id', required=False, help='User ID that owns the pipeline.')
@click.option('--app_id', required=False, help='App ID that contains the pipeline.')
@click.option('--pipeline_id', required=False, help='Pipeline ID.')
@click.option('--pipeline_version_id', required=False, help='Pipeline Version ID.')
@click.option(
    '--timeout',
    type=int,
    default=3600,
    help='Maximum time to wait for completion in seconds. Default 3600 (1 hour).',
)
@click.option(
    '--monitor_interval',
    type=int,
    default=10,
    help='Interval between status checks in seconds. Default 10.',
)
@click.option(
    '--log_file',
    type=click.Path(),
    required=False,
    help='Path to file where logs should be written. If not provided, logs are displayed on console.',
)
@click.pass_context
def monitor(
    ctx,
    pipeline_version_run_id,
    pipeline_version_run_id_flag,
    user_id,
    app_id,
    pipeline_id,
    pipeline_version_id,
    timeout,
    monitor_interval,
    log_file,
):
    """Monitor an existing pipeline version run.

    Monitor the current status and logs of a running pipeline.

    Examples:

        # Using positional argument
        clarifai pr monitor <pipeline_version_run_id>

        # Using flag
        clarifai pipelinerun monitor --pipeline_version_run_id=<id>

        # With explicit parameters
        clarifai pr monitor <pipeline_version_run_id> \\
            --user_id=USER_ID \\
            --app_id=APP_ID \\
            --pipeline_id=PIPELINE_ID \\
            --pipeline_version_id=VERSION_ID

        # With custom timeout and interval
        clarifai pr monitor <pipeline_version_run_id> \\
            --timeout=7200 \\
            --monitor_interval=5
    """
    import json

    validate_context(ctx)

    # Resolve pipeline_version_run_id from positional or flag
    run_id = pipeline_version_run_id or pipeline_version_run_id_flag
    if not run_id:
        raise click.UsageError(
            "pipeline_version_run_id is required. "
            "Provide it as a positional argument or use --pipeline_version_run_id flag."
        )

    # Load parameters from config-lock.yaml if not provided
    user_id, app_id, pipeline_id, pipeline_version_id = _load_pipeline_params_from_config(
        user_id, app_id, pipeline_id, pipeline_version_id
    )

    # Validate required parameters
    _validate_pipeline_params(user_id, app_id, pipeline_id, pipeline_version_id)

    # Create Pipeline object
    pipeline = _create_pipeline(ctx, user_id, app_id, pipeline_id, pipeline_version_id)

    # Set the pipeline_version_run_id for monitoring
    pipeline.pipeline_version_run_id = run_id

    # Set log file if provided
    if log_file:
        pipeline.log_file = log_file

    # Monitor the pipeline run
    try:
        result = pipeline.monitor_only(timeout=timeout, monitor_interval=monitor_interval)
        click.echo(json.dumps(result, indent=2, default=str))
    except Exception as e:
        logger.error(f"Failed to monitor pipeline version run: {e}")
        raise click.ClickException(str(e))
