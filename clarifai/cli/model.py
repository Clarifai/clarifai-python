import click
from clarifai.cli.base import cli


@cli.group(['model'])
def model():
  """Manage models: upload, test locally, run_locally, predict"""
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
@click.option(
    '--mode',
    type=click.Choice(['env', 'container'], case_sensitive=False),
    default='env',
    show_default=True,
    help=
    'Specify how to test the model locally: "env" for virtual environment or "container" for Docker container. Defaults to "env".'
)
@click.option(
    '--keep_env',
    is_flag=True,
    help=
    'Keep the virtual environment after testing the model locally (applicable for virtualenv mode). Defaults to False.'
)
@click.option(
    '--keep_image',
    is_flag=True,
    help=
    'Keep the Docker image after testing the model locally (applicable for container mode). Defaults to False.'
)
def test_locally(model_path, keep_env=False, keep_image=False, mode='env'):
  """Test model locally."""
  try:
    from clarifai.runners.models import model_run_locally
    if mode == "env":
      click.echo("Testing model locally in a virtual environment...")
      model_run_locally.main(model_path, run_model_server=False, keep_env=keep_env)
    elif mode == "container":
      click.echo("Testing model locally inside a container...")
      model_run_locally.main(
          model_path, inside_container=True, run_model_server=False, keep_image=keep_image)
    click.echo("Model tested su")
  except Exception as e:
    click.echo(f"Failed to test model locally: {e}", err=True)


@model.command()
@click.option(
    '--model_path',
    type=click.Path(exists=True),
    required=True,
    help='Path to the model directory.')
@click.option(
    '--port',
    '-p',
    type=int,
    default=8000,
    show_default=True,
    help="The port to host the gRPC server for running the model locally. Defaults to 8000.")
@click.option(
    '--mode',
    type=click.Choice(['env', 'container'], case_sensitive=False),
    default='env',
    show_default=True,
    help=
    'Specifies how to run the model: "env" for virtual environment or "container" for Docker container. Defaults to "env".'
)
@click.option(
    '--keep_env',
    is_flag=True,
    help=
    'Keep the virtual environment after testing the model locally (applicable for virtualenv mode). Defaults to False.'
)
@click.option(
    '--keep_image',
    is_flag=True,
    help=
    'Keep the Docker image after testing the model locally (applicable for container mode). Defaults to False.'
)
def run_locally(model_path, port, mode, keep_env, keep_image):
  """Run the model locally and start a gRPC server to serve the model."""
  try:
    from clarifai.runners.models import model_run_locally

    if mode == "virtualenv":
      click.echo("Running model locally in a virtual environment...")
      model_run_locally.main(model_path, run_model_server=True, keep_env=keep_env, port=port)
    elif mode == "container":
      click.echo("Running model locally inside a container...")
      model_run_locally.main(
          model_path,
          inside_container=True,
          run_model_server=True,
          port=port,
          keep_image=keep_image)
    click.echo(f"Model server started locally from {model_path} in {mode} mode.")
  except Exception as e:
    click.echo(f"Failed to start the model server locally: {e}", err=True)
