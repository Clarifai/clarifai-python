import os
import shutil

import click

from clarifai.cli.base import cli
from clarifai.utils.cli import (
    AliasedGroup,
    convert_timestamp_to_string,
    display_co_resources,
    validate_context,
)
from clarifai.utils.logging import logger


@cli.group(
    ['pipelinestep', 'ps'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def pipeline_step():
    """Manage pipeline steps."""


@pipeline_step.command()
@click.argument("pipeline_step_path", type=click.Path(exists=True), required=False, default=".")
@click.option(
    '--skip_dockerfile',
    is_flag=True,
    help='Flag to skip generating a dockerfile so that you can manually edit an already created dockerfile.',
)
def upload(pipeline_step_path, skip_dockerfile):
    """Upload a pipeline step to Clarifai.

    PIPELINE_STEP_PATH: Path to the pipeline step directory. If not specified, the current directory is used by default.
    """
    from clarifai.runners.pipeline_steps.pipeline_step_builder import upload_pipeline_step

    upload_pipeline_step(pipeline_step_path, skip_dockerfile)


@pipeline_step.command()
@click.argument(
    "pipeline_step_path",
    type=click.Path(),
    required=False,
    default=".",
)
def init(pipeline_step_path):
    """Initialize a new pipeline step directory structure.

    Creates the following structure in the specified directory:
    ├── 1/
    │   └── pipeline_step.py
    ├── requirements.txt
    └── config.yaml

    PIPELINE_STEP_PATH: Path where to create the pipeline step directory structure. If not specified, the current directory is used by default.
    """
    from clarifai.cli.templates.pipeline_step_templates import (
        get_config_template,
        get_pipeline_step_template,
        get_requirements_template,
    )

    # Resolve the absolute path
    pipeline_step_path = os.path.abspath(pipeline_step_path)

    # Create the pipeline step directory if it doesn't exist
    os.makedirs(pipeline_step_path, exist_ok=True)

    # Create the 1/ subdirectory
    pipeline_step_version_dir = os.path.join(pipeline_step_path, "1")
    os.makedirs(pipeline_step_version_dir, exist_ok=True)

    # Create pipeline_step.py
    pipeline_step_py_path = os.path.join(pipeline_step_version_dir, "pipeline_step.py")
    if os.path.exists(pipeline_step_py_path):
        logger.warning(f"File {pipeline_step_py_path} already exists, skipping...")
    else:
        pipeline_step_template = get_pipeline_step_template()
        with open(pipeline_step_py_path, 'w') as f:
            f.write(pipeline_step_template)
        logger.info(f"Created {pipeline_step_py_path}")

    # Create requirements.txt
    requirements_path = os.path.join(pipeline_step_path, "requirements.txt")
    if os.path.exists(requirements_path):
        logger.warning(f"File {requirements_path} already exists, skipping...")
    else:
        requirements_template = get_requirements_template()
        with open(requirements_path, 'w') as f:
            f.write(requirements_template)
        logger.info(f"Created {requirements_path}")

    # Create config.yaml
    config_path = os.path.join(pipeline_step_path, "config.yaml")
    if os.path.exists(config_path):
        logger.warning(f"File {config_path} already exists, skipping...")
    else:
        config_template = get_config_template()
        with open(config_path, 'w') as f:
            f.write(config_template)
        logger.info(f"Created {config_path}")

    logger.info(f"Pipeline step initialization complete in {pipeline_step_path}")
    logger.info("Next steps:")
    logger.info("1. Search for '# TODO: please fill in' comments in the generated files")
    logger.info("2. Update the pipeline step configuration in config.yaml")
    logger.info("3. Add your pipeline step dependencies to requirements.txt")
    logger.info("4. Implement your pipeline step logic in 1/pipeline_step.py")


@pipeline_step.command(['local-run'])
@click.argument("pipeline_step_path", type=click.Path(exists=True), required=False, default=".")
@click.option(
    '--mode',
    type=click.Choice(['container']),
    default='container',
    help='Execution mode. Currently only "container" is supported.',
)
@click.option(
    '--keep-image',
    is_flag=True,
    default=False,
    help='Keep the Docker image after the pipeline step finishes.',
)
@click.option(
    '--step-args',
    default=None,
    help='Arguments to pass to pipeline_step.py (e.g., "--param_a hello --param_b world").',
)
def local_run(pipeline_step_path, mode, keep_image, step_args):
    """Run a pipeline step locally in a Docker container.

    \b
    PIPELINE_STEP_PATH: Path to the pipeline step directory (containing config.yaml,
    requirements.txt, and 1/pipeline_step.py). Defaults to current directory.

    \b
    Pass arguments to the step script via --step-args:
      clarifai pipelinestep local-run ./my-step --step-args "--param_a hello --param_b world"

    This reuses the same Docker build infrastructure as ``clarifai model serve
    --mode container`` but executes pipeline_step.py once and exits.
    """
    # Parse step-args string into a list
    if step_args:
        import shlex

        step_args = shlex.split(step_args)
    from clarifai.runners.pipeline_steps.pipeline_run_locally import PipelineStepRunLocally

    manager = PipelineStepRunLocally(pipeline_step_path)

    if not manager.is_docker_installed():
        raise click.ClickException("Docker is not installed.")

    # Generate Dockerfile if missing
    manager.builder.create_dockerfile()

    image_tag = manager._docker_hash()
    step_id = manager.config['pipeline_step']['id'].lower()
    image_name = f"{step_id}:{image_tag}"
    container_name = f"{step_id}-local-run"

    if not manager.docker_image_exists(image_name):
        logger.info("Building Docker image...")
        manager.build_docker_image(image_name=image_name)

    try:
        manager.run_pipeline_step_container(
            image_name=image_name,
            container_name=container_name,
            step_args=step_args,
        )
    finally:
        if manager.container_exists(container_name):
            manager.stop_docker_container(container_name)
            manager.remove_docker_container(container_name)
        if not keep_image:
            manager.remove_docker_image(image_name)


@pipeline_step.command(['ls'])
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=16)
@click.option(
    '--app_id',
    required=True,
    help='App ID to list pipeline steps from.',
)
@click.option(
    '--user_id',
    required=False,
    help='User ID to list pipeline steps from. If not provided, uses current user.',
)
@click.option(
    '--pipeline_id',
    required=False,
    help='Pipeline ID to list pipeline steps from.',
)
@click.pass_context
def list(ctx, page_no, per_page, app_id, user_id, pipeline_id):
    """List all pipeline steps for the user."""
    validate_context(ctx)

    target_user_id = user_id or ctx.obj.current.user_id

    from clarifai.client.app import App

    app = App(
        app_id=app_id,
        user_id=target_user_id,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
    )
    response = app.list_pipeline_steps(pipeline_id=pipeline_id, page_no=page_no, per_page=per_page)

    display_co_resources(
        response,
        custom_columns={
            'PIPELINE_STEP_ID': lambda ps: getattr(ps, 'pipeline_step_id', ''),
            'USER_ID': lambda ps: getattr(ps, 'user_id', ''),
            'APP_ID': lambda ps: getattr(ps, 'app_id', ''),
            'VERSION_ID': lambda ps: getattr(ps, 'pipeline_step_version_id', ''),
            'VISIBILITY': lambda ps: getattr(ps, 'visibility', ''),
            'DESCRIPTION': lambda ps: getattr(ps, 'description', ''),
            'CREATED_AT': lambda ps: convert_timestamp_to_string(getattr(ps, 'created_at', '')),
            'MODIFIED_AT': lambda ps: convert_timestamp_to_string(getattr(ps, 'modified_at', '')),
        },
        sort_by_columns=[
            ('CREATED_AT', 'desc'),
            ('PIPELINE_STEP_ID', 'asc'),
        ],
    )
