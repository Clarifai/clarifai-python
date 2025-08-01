import os
import shutil

import click

from clarifai.cli.base import cli
from clarifai.client.app import App
from clarifai.client.user import User
from clarifai.utils.cli import (
    AliasedGroup,
    convert_timestamp_to_string,
    display_co_resources,
    validate_context,
)
from clarifai.utils.logging import logger


@cli.group(
    ['pipeline', 'pl'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def pipeline():
    """Manage pipelines: upload, init, list, etc"""


@pipeline.command()
@click.argument("path", type=click.Path(exists=True), required=False, default=".")
def upload(path):
    """Upload a pipeline with associated pipeline steps to Clarifai.

    PATH: Path to the pipeline configuration file or directory containing config.yaml. If not specified, the current directory is used by default.
    """
    from clarifai.runners.pipelines.pipeline_builder import upload_pipeline

    upload_pipeline(path)


@pipeline.command()
@click.option(
    '--config',
    type=click.Path(exists=True),
    required=False,
    help='Path to the pipeline run config file.',
)
@click.option('--pipeline_id', required=False, help='Pipeline ID to run.')
@click.option('--pipeline_version_id', required=False, help='Pipeline Version ID to run.')
@click.option(
    '--pipeline_version_run_id',
    required=False,
    help='Pipeline Version Run ID. If not provided, a UUID will be generated.',
)
@click.option('--user_id', required=False, help='User ID of the pipeline.')
@click.option('--app_id', required=False, help='App ID that contains the pipeline.')
@click.option('--nodepool_id', required=False, help='Nodepool ID to run the pipeline on.')
@click.option(
    '--compute_cluster_id', required=False, help='Compute Cluster ID to run the pipeline on.'
)
@click.option('--pipeline_url', required=False, help='Pipeline URL to run.')
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
@click.option(
    '--monitor',
    is_flag=True,
    default=False,
    help='Monitor an existing pipeline run instead of starting a new one. Requires pipeline_version_run_id.',
)
@click.pass_context
def run(
    ctx,
    config,
    pipeline_id,
    pipeline_version_id,
    pipeline_version_run_id,
    user_id,
    app_id,
    nodepool_id,
    compute_cluster_id,
    pipeline_url,
    timeout,
    monitor_interval,
    log_file,
    monitor,
):
    """Run a pipeline and monitor its progress."""
    import json

    from clarifai.client.pipeline import Pipeline
    from clarifai.utils.cli import from_yaml, validate_context

    validate_context(ctx)

    if config:
        config_data = from_yaml(config)
        pipeline_id = config_data.get('pipeline_id', pipeline_id)
        pipeline_version_id = config_data.get('pipeline_version_id', pipeline_version_id)
        pipeline_version_run_id = config_data.get(
            'pipeline_version_run_id', pipeline_version_run_id
        )
        user_id = config_data.get('user_id', user_id)
        app_id = config_data.get('app_id', app_id)
        nodepool_id = config_data.get('nodepool_id', nodepool_id)
        compute_cluster_id = config_data.get('compute_cluster_id', compute_cluster_id)
        pipeline_url = config_data.get('pipeline_url', pipeline_url)
        timeout = config_data.get('timeout', timeout)
        monitor_interval = config_data.get('monitor_interval', monitor_interval)
        log_file = config_data.get('log_file', log_file)
        monitor = config_data.get('monitor', monitor)
    elif ctx.obj.current:
        if not user_id:
            user_id = ctx.obj.current.get('user_id', '')
        if not app_id:
            app_id = ctx.obj.current.get('app_id', '')
        if not pipeline_id:
            pipeline_id = ctx.obj.current.get('pipeline_id', '')
        if not pipeline_version_id:
            pipeline_version_id = ctx.obj.current.get('pipeline_version_id', '')
        if not nodepool_id:
            nodepool_id = ctx.obj.current.get('nodepool_id', '')
        if not compute_cluster_id:
            compute_cluster_id = ctx.obj.current.get('compute_cluster_id', '')

    # compute_cluster_id and nodepool_id are mandatory regardless of whether pipeline_url is provided
    if not compute_cluster_id or not nodepool_id:
        raise ValueError("--compute_cluster_id and --nodepool_id are mandatory parameters.")

    # When monitor flag is used, pipeline_version_run_id is mandatory
    if monitor and not pipeline_version_run_id:
        raise ValueError("--pipeline_version_run_id is required when using --monitor flag.")

    if pipeline_url:
        # When using pipeline_url, other parameters are optional (will be parsed from URL)
        required_params_provided = True
    else:
        # When not using pipeline_url, all individual parameters are required
        required_params_provided = all([pipeline_id, user_id, app_id, pipeline_version_id])

    if not required_params_provided:
        raise ValueError(
            "Either --user_id & --app_id & --pipeline_id & --pipeline_version_id or --pipeline_url must be provided."
        )

    if pipeline_url:
        pipeline = Pipeline(
            url=pipeline_url,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
            pipeline_version_run_id=pipeline_version_run_id,
            nodepool_id=nodepool_id,
            compute_cluster_id=compute_cluster_id,
            log_file=log_file,
        )
    else:
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            pipeline_version_id=pipeline_version_id,
            pipeline_version_run_id=pipeline_version_run_id,
            user_id=user_id,
            app_id=app_id,
            nodepool_id=nodepool_id,
            compute_cluster_id=compute_cluster_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
            log_file=log_file,
        )

    if monitor:
        # Monitor existing pipeline run instead of starting new one
        result = pipeline.monitor_only(timeout=timeout, monitor_interval=monitor_interval)
    else:
        # Start new pipeline run and monitor it
        result = pipeline.run(timeout=timeout, monitor_interval=monitor_interval)
    click.echo(json.dumps(result, indent=2, default=str))


@pipeline.command()
@click.argument(
    "pipeline_path",
    type=click.Path(),
    required=False,
    default=".",
)
def init(pipeline_path):
    """Initialize a new pipeline project structure.

    Creates the following structure in the specified directory:
    ├── config.yaml          # Pipeline configuration
    ├── stepA/               # First pipeline step
    │   ├── config.yaml     # Step A configuration
    │   ├── requirements.txt # Step A dependencies
    │   └── 1/
    │       └── pipeline_step.py  # Step A implementation
    ├── stepB/               # Second pipeline step
    │   ├── config.yaml     # Step B configuration
    │   ├── requirements.txt # Step B dependencies
    │   └── 1/
    │       └── pipeline_step.py  # Step B implementation
    └── README.md           # Documentation

    PIPELINE_PATH: Path where to create the pipeline project structure. If not specified, the current directory is used by default.
    """
    from clarifai.cli.templates.pipeline_templates import (
        get_pipeline_config_template,
        get_pipeline_step_config_template,
        get_pipeline_step_requirements_template,
        get_pipeline_step_template,
        get_readme_template,
    )

    # Resolve the absolute path
    pipeline_path = os.path.abspath(pipeline_path)

    # Create the pipeline directory if it doesn't exist
    os.makedirs(pipeline_path, exist_ok=True)

    # Create pipeline config.yaml
    config_path = os.path.join(pipeline_path, "config.yaml")
    if os.path.exists(config_path):
        logger.warning(f"File {config_path} already exists, skipping...")
    else:
        config_template = get_pipeline_config_template()
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_template)
        logger.info(f"Created {config_path}")

    # Create README.md
    readme_path = os.path.join(pipeline_path, "README.md")
    if os.path.exists(readme_path):
        logger.warning(f"File {readme_path} already exists, skipping...")
    else:
        readme_template = get_readme_template()
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_template)
        logger.info(f"Created {readme_path}")

    # Create pipeline steps (stepA and stepB)
    for step_id in ["stepA", "stepB"]:
        step_dir = os.path.join(pipeline_path, step_id)
        os.makedirs(step_dir, exist_ok=True)

        # Create the 1/ subdirectory for the step version
        step_version_dir = os.path.join(step_dir, "1")
        os.makedirs(step_version_dir, exist_ok=True)

        # Create step config.yaml
        step_config_path = os.path.join(step_dir, "config.yaml")
        if os.path.exists(step_config_path):
            logger.warning(f"File {step_config_path} already exists, skipping...")
        else:
            step_config_template = get_pipeline_step_config_template(step_id)
            with open(step_config_path, 'w', encoding='utf-8') as f:
                f.write(step_config_template)
            logger.info(f"Created {step_config_path}")

        # Create step requirements.txt
        step_requirements_path = os.path.join(step_dir, "requirements.txt")
        if os.path.exists(step_requirements_path):
            logger.warning(f"File {step_requirements_path} already exists, skipping...")
        else:
            step_requirements_template = get_pipeline_step_requirements_template()
            with open(step_requirements_path, 'w', encoding='utf-8') as f:
                f.write(step_requirements_template)
            logger.info(f"Created {step_requirements_path}")

        # Create step pipeline_step.py
        step_py_path = os.path.join(step_version_dir, "pipeline_step.py")
        if os.path.exists(step_py_path):
            logger.warning(f"File {step_py_path} already exists, skipping...")
        else:
            step_py_template = get_pipeline_step_template(step_id)
            with open(step_py_path, 'w', encoding='utf-8') as f:
                f.write(step_py_template)
            logger.info(f"Created {step_py_path}")

    logger.info(f"Pipeline initialization complete in {pipeline_path}")
    logger.info("Next steps:")
    logger.info("1. Search for '# TODO: please fill in' comments in the generated files")
    logger.info("2. Update your user_id and app_id in all config.yaml files")
    logger.info(
        "3. Implement your pipeline step logic in stepA/1/pipeline_step.py and stepB/1/pipeline_step.py"
    )
    logger.info("4. Add dependencies to requirements.txt files as needed")
    logger.info("5. Run 'clarifai pipeline upload config.yaml' to upload your pipeline")


@pipeline.command(['ls'])
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=16)
@click.option(
    '--app_id',
    required=False,
    help='App ID to list pipelines from. If not provided, lists across all apps.',
)
@click.pass_context
def list(ctx, page_no, per_page, app_id):
    """List all pipelines for the user."""
    validate_context(ctx)

    if app_id:
        app = App(
            app_id=app_id,
            user_id=ctx.obj.current.user_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
        )
        response = app.list_pipelines(page_no=page_no, per_page=per_page)
    else:
        user = User(
            user_id=ctx.obj.current.user_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
        )
        response = user.list_pipelines(page_no=page_no, per_page=per_page)

    display_co_resources(
        response,
        custom_columns={
            'ID': lambda p: getattr(p, 'pipeline_id', ''),
            'USER_ID': lambda p: getattr(p, 'user_id', ''),
            'APP_ID': lambda p: getattr(p, 'app_id', ''),
            'VERSION_ID': lambda p: getattr(p, 'pipeline_version_id', ''),
            'DESCRIPTION': lambda p: getattr(p, 'description', ''),
            'CREATED_AT': lambda ps: convert_timestamp_to_string(getattr(ps, 'created_at', '')),
            'MODIFIED_AT': lambda ps: convert_timestamp_to_string(getattr(ps, 'modified_at', '')),
        },
        sort_by_columns=[
            ('CREATED_AT', 'desc'),
            ('ID', 'asc'),
        ],
    )
