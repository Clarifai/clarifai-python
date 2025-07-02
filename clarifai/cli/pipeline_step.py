import os
import shutil

import click

from clarifai.cli.base import cli
from clarifai.utils.logging import logger


@cli.group(
    ['pipelinestep', 'ps'],
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def pipeline_step():
    """Manage pipeline steps: upload, test, etc"""


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
