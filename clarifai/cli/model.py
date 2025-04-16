import click

from clarifai.cli.base import cli


@cli.group(['model'])
def model():
  """Manage models: upload, test, local dev, predict, etc"""


@model.command()
@click.argument("model_path", type=click.Path(exists=True), required=False, default=".")
@click.option(
    '--stage',
    required=False,
    type=click.Choice(['runtime', 'build', 'upload'], case_sensitive=True),
    default="upload",
    show_default=True,
    help=
    'The stage we are calling download checkpoints from. Typically this would "upload" and will download checkpoints if config.yaml checkpoints section has when set to "upload". Other options include "runtime" to be used in load_model or "upload" to be used during model upload. Set this stage to whatever you have in config.yaml to force downloading now.'
)
@click.option(
    '--skip_dockerfile',
    is_flag=True,
    help=
    'Flag to skip generating a dockerfile so that you can manually edit an already created dockerfile.',
)
def upload(model_path, stage, skip_dockerfile):
  """Upload a model to Clarifai.

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
  """
  from clarifai.runners.models.model_builder import upload_model
  upload_model(model_path, stage, skip_dockerfile)


@model.command()
@click.argument(
    "model_path",
    type=click.Path(exists=True),
    required=False,
    default=".",
)
@click.option(
    '--out_path',
    type=click.Path(exists=False),
    required=False,
    default=None,
    help=
    'Option path to write the checkpoints to. This will place them in {out_path}/1/checkpoints If not provided it will default to {model_path}/1/checkpoints where the config.yaml is read.'
)
@click.option(
    '--stage',
    required=False,
    type=click.Choice(['runtime', 'build', 'upload'], case_sensitive=True),
    default="build",
    show_default=True,
    help=
    'The stage we are calling download checkpoints from. Typically this would be in the build stage which is the default. Other options include "runtime" to be used in load_model or "upload" to be used during model upload. Set this stage to whatever you have in config.yaml to force downloading now.'
)
def download_checkpoints(model_path, out_path, stage):
  """Download checkpoints from external source to local model_path

  MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
  """

  from clarifai.runners.models.model_builder import ModelBuilder
  builder = ModelBuilder(model_path, download_validation_only=True)
  builder.download_checkpoints(stage=stage, checkpoint_path_override=out_path)


@model.command()
@click.argument(
    "model_path",
    type=click.Path(exists=True),
    required=False,
    default=".",
)
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
@click.option(
    '--skip_dockerfile',
    is_flag=True,
    help=
    'Flag to skip generating a dockerfile so that you can manually edit an already created dockerfile. Apply for `--mode conatainer`.',
)
def test_locally(model_path, keep_env=False, keep_image=False, mode='env', skip_dockerfile=False):
  """Test model locally."""
  try:
    from clarifai.runners.models import model_run_locally
    if mode == 'env' and keep_image:
      raise ValueError("'keep_image' is applicable only for 'container' mode")
    if mode == 'container' and keep_env:
      raise ValueError("'keep_env' is applicable only for 'env' mode")

    if mode == "env":
      click.echo("Testing model locally in a virtual environment...")
      model_run_locally.main(model_path, run_model_server=False, keep_env=keep_env)
    elif mode == "container":
      click.echo("Testing model locally inside a container...")
      model_run_locally.main(
          model_path,
          inside_container=True,
          run_model_server=False,
          keep_image=keep_image,
          skip_dockerfile=skip_dockerfile)
    click.echo("Model tested successfully.")
  except Exception as e:
    click.echo(f"Failed to test model locally: {e}", err=True)


@model.command()
@click.argument(
    "model_path",
    type=click.Path(exists=True),
    required=False,
    default=".",
)
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
@click.option(
    '--skip_dockerfile',
    is_flag=True,
    help=
    'Flag to skip generating a dockerfile so that you can manually edit an already created dockerfile. Apply for `--mode conatainer`.',
)
def run_locally(model_path, port, mode, keep_env, keep_image, skip_dockerfile=False):
  """Run the model locally and start a gRPC server to serve the model."""
  try:
    from clarifai.runners.models import model_run_locally
    if mode == 'env' and keep_image:
      raise ValueError("'keep_image' is applicable only for 'container' mode")
    if mode == 'container' and keep_env:
      raise ValueError("'keep_env' is applicable only for 'env' mode")

    if mode == "env":
      click.echo("Running model locally in a virtual environment...")
      model_run_locally.main(model_path, run_model_server=True, keep_env=keep_env, port=port)
    elif mode == "container":
      click.echo("Running model locally inside a container...")
      model_run_locally.main(
          model_path,
          inside_container=True,
          run_model_server=True,
          port=port,
          keep_image=keep_image,
          skip_dockerfile=skip_dockerfile)
    click.echo(f"Model server started locally from {model_path} in {mode} mode.")
  except Exception as e:
    click.echo(f"Failed to starts model server locally: {e}", err=True)


@model.command()
@click.argument(
    "model_path",
    type=click.Path(exists=True),
    required=False,
    default=".",
)
def local_dev(model_path):
  """Run the model as a local dev runner to help debug your model connected to the API. You must set several envvars such as CLARIFAI_PAT, CLARIFAI_RUNNER_ID, CLARIFAI_NODEPOOL_ID, CLARIFAI_COMPUTE_CLUSTER_ID.

  MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
  """
  from clarifai.runners.server import serve
  serve(model_path)


@model.command()
@click.option(
    '--config',
    type=click.Path(exists=True),
    required=False,
    help='Path to the model predict config file.')
@click.option('--model_id', required=False, help='Model ID of the model used to predict.')
@click.option('--user_id', required=False, help='User ID of the model used to predict.')
@click.option('--app_id', required=False, help='App ID of the model used to predict.')
@click.option('--model_url', required=False, help='Model URL of the model used to predict.')
@click.option('--file_path', required=False, help='File path of file for the model to predict')
@click.option('--url', required=False, help='URL to the file for the model to predict')
@click.option('--bytes', required=False, help='Bytes to the file for the model to predict')
@click.option('--input_type', required=False, help='Type of input')
@click.option(
    '-cc_id',
    '--compute_cluster_id',
    required=False,
    help='Compute Cluster ID to use for the model')
@click.option('-np_id', '--nodepool_id', required=False, help='Nodepool ID to use for the model')
@click.option(
    '-dpl_id', '--deployment_id', required=False, help='Deployment ID to use for the model')
@click.option(
    '--inference_params', required=False, default='{}', help='Inference parameters to override')
@click.option('--output_config', required=False, default='{}', help='Output config to override')
@click.pass_context
def predict(ctx, config, model_id, user_id, app_id, model_url, file_path, url, bytes, input_type,
            compute_cluster_id, nodepool_id, deployment_id, inference_params, output_config):
  """Predict using the given model"""
  import json

  from clarifai.client.model import Model
  from clarifai.utils.cli import from_yaml, validate_context
  validate_context(ctx)
  if config:
    config = from_yaml(config)
    model_id, user_id, app_id, model_url, file_path, url, bytes, input_type, compute_cluster_id, nodepool_id, deployment_id, inference_params, output_config = (
        config.get(k, v)
        for k, v in [('model_id', model_id), ('user_id', user_id), ('app_id', app_id), (
            'model_url', model_url), ('file_path', file_path), ('url', url), ('bytes', bytes), (
                'input_type', input_type), ('compute_cluster_id', compute_cluster_id), (
                    'nodepool_id',
                    nodepool_id), ('deployment_id',
                                   deployment_id), ('inference_params',
                                                    inference_params), ('output_config',
                                                                        output_config)])
  if sum([opt[1] for opt in [(model_id, 1), (user_id, 1), (app_id, 1), (model_url, 3)]
          if opt[0]]) != 3:
    raise ValueError("Either --model_id & --user_id & --app_id or --model_url must be provided.")
  if compute_cluster_id or nodepool_id or deployment_id:
    if sum([
        opt[1] for opt in [(compute_cluster_id, 0.5), (nodepool_id, 0.5), (deployment_id, 1)]
        if opt[0]
    ]) != 1:
      raise ValueError(
          "Either --compute_cluster_id & --nodepool_id or --deployment_id must be provided.")
  if model_url:
    model = Model(url=model_url, pat=ctx.obj['pat'], base_url=ctx.obj['base_url'])
  else:
    model = Model(
        model_id=model_id,
        user_id=user_id,
        app_id=app_id,
        pat=ctx.obj['pat'],
        base_url=ctx.obj['base_url'])

  if inference_params:
    inference_params = json.loads(inference_params)
  if output_config:
    output_config = json.loads(output_config)

  if file_path:
    model_prediction = model.predict_by_filepath(
        filepath=file_path,
        input_type=input_type,
        compute_cluster_id=compute_cluster_id,
        nodepool_id=nodepool_id,
        deployment_id=deployment_id,
        inference_params=inference_params,
        output_config=output_config)
  elif url:
    model_prediction = model.predict_by_url(
        url=url,
        input_type=input_type,
        compute_cluster_id=compute_cluster_id,
        nodepool_id=nodepool_id,
        deployment_id=deployment_id,
        inference_params=inference_params,
        output_config=output_config)
  elif bytes:
    bytes = str.encode(bytes)
    model_prediction = model.predict_by_bytes(
        input_bytes=bytes,
        input_type=input_type,
        compute_cluster_id=compute_cluster_id,
        nodepool_id=nodepool_id,
        deployment_id=deployment_id,
        inference_params=inference_params,
        output_config=output_config)  ## TO DO: Add support for input_id
  click.echo(model_prediction)
