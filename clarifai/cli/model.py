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
    '--keep_env',
    is_flag=True,
    help='Flag to keep the virtual environment after testing the model\
    locally. Defaults to False, which will delete the virtual environment after testing.')
def test_locally(model_path, keep_env=False):
  """Test model locally."""
  try:
    from clarifai.runners.models import model_run_locally
    model_run_locally.main(model_path, keep_env=keep_env)
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
@click.option(
    '--input_id', required=False, help='Existing input id in the app for the model to predict')
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
def predict(ctx, config, model_id, user_id, app_id, model_url, file_path, url, bytes, input_id,
            input_type, compute_cluster_id, nodepool_id, deployment_id, inference_params,
            output_config):
  """Predict using the given model"""
  import json

  from clarifai.client.deployment import Deployment
  from clarifai.client.input import Input
  from clarifai.client.model import Model
  from clarifai.client.nodepool import Nodepool
  from clarifai.utils.cli import from_yaml
  if config:
    config = from_yaml(config)
    model_id, user_id, app_id, model_url, file_path, url, bytes, input_id, input_type, compute_cluster_id, nodepool_id, deployment_id, inference_params, output_config = (
        config.get(k, v)
        for k, v in [('model_id', model_id), ('user_id', user_id), ('app_id', app_id), (
            'model_url', model_url), ('file_path', file_path), ('url', url), ('bytes', bytes), (
                'input_id',
                input_id), ('input_type',
                            input_type), ('compute_cluster_id',
                                          compute_cluster_id), ('nodepool_id', nodepool_id), (
                                              'deployment_id',
                                              deployment_id), ('inference_params',
                                                               inference_params), ('output_config',
                                                                                   output_config)])
  if sum([opt[1] for opt in [(model_id, 1), (user_id, 1), (app_id, 1), (model_url, 3)]
          if opt[0]]) != 3:
    raise ValueError("Either --model_id & --user_id & --app_id or --model_url must be provided.")
  if sum([1 for opt in [file_path, url, bytes, input_id] if opt]) != 1:
    raise ValueError("Exactly one of --file_path, --url, --bytes or --input_id must be provided.")
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
        output_config=output_config)
  elif input_id:
    inputs = [Input.get_input(input_id)]
    runner_selector = None
    if deployment_id:
      runner_selector = Deployment.get_runner_selector(
          user_id=ctx.obj['user_id'], deployment_id=deployment_id)
    elif compute_cluster_id and nodepool_id:
      runner_selector = Nodepool.get_runner_selector(
          user_id=ctx.obj['user_id'],
          compute_cluster_id=compute_cluster_id,
          nodepool_id=nodepool_id)
    model_prediction = model.predict(
        inputs=inputs,
        runner_selector=runner_selector,
        inference_params=inference_params,
        output_config=output_config)
  click.echo(model_prediction)
