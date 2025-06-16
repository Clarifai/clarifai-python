import shutil

import click

from clarifai.cli.base import cli


@cli.group(
    ['pipeline'],
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def pipeline():
    """Manage pipelines: upload, etc"""


@pipeline.command()
@click.argument("config_path", type=click.Path(exists=True), required=False, default="config.yaml")
def upload(config_path):
    """Upload a pipeline with associated pipeline steps to Clarifai.

    CONFIG_PATH: Path to the pipeline configuration file. If not specified, 'config.yaml' in the current directory is used by default.
    """
    from clarifai.runners.pipelines.pipeline_builder import upload_pipeline

    upload_pipeline(config_path)