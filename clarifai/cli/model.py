import click
from clarifai.cli.base import cli


@cli.group(['model'])
def model():
  """Manage models: upload, test locally"""
  pass


@model.command()
@click.option(
    '--model_path',
    type=click.Path(exists=True),
    required=True,
    help='Path to the model directory.')
@click.option(
    '--download_checkpoints',
    is_flag=True,
    help=
    'Flag to download checkpoints before uploading and including them in the tar file that is uploaded. Defaults to False, which will attempt to download them at docker build time.',
)
@click.option(
    '--skip_dockerfile',
    is_flag=True,
    help=
    'Flag to skip generating a dockerfile so that you can manually edit an already created dockerfile.',
)
def upload(model_path, download_checkpoints, skip_dockerfile):
  """Upload a model to Clarifai."""
  from clarifai.runners.models import model_upload

  model_upload.main(model_path, download_checkpoints, skip_dockerfile)


@model.command()
@click.option(
    '--model_path',
    type=click.Path(exists=True),
    required=True,
    help='Path to the model directory.')
def test_locally(model_path):
  """Test model locally."""
  try:
    from clarifai.runners.models import model_run_locally
    model_run_locally.main(model_path)
    click.echo(f"Model tested locally from {model_path}.")
  except Exception as e:
    click.echo(f"Failed to test model locally: {e}", err=True)


@model.command()
@click.option(
    '--model_path',
    type=click.Path(exists=True),
    required=True,
    help='Path to the model directory.')
def run_locally(model_path):
  """Run model locally and starts a GRPC server to serve the model."""
  try:
    from clarifai.runners.models import model_run_locally
    model_run_locally.main(model_path, run_model_server=True)
    click.echo(f"Model server started locally from {model_path}.")
  except Exception as e:
    click.echo(f"Failed to starts model server locally: {e}", err=True)
