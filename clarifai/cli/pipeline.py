import os
import shutil

import click

from clarifai.cli.base import cli
from clarifai.utils.logging import logger


@cli.group(
    ['pipeline', 'pl'],
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def pipeline():
    """Manage pipelines: upload, init, etc"""


@pipeline.command()
@click.argument("path", type=click.Path(exists=True), required=False, default=".")
def upload(path):
    """Upload a pipeline with associated pipeline steps to Clarifai.

    PATH: Path to the pipeline configuration file or directory containing config.yaml. If not specified, the current directory is used by default.
    """
    from clarifai.runners.pipelines.pipeline_builder import upload_pipeline

    upload_pipeline(path)


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
