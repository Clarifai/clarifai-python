import click
from clarifai.cli.base import cli
from clarifai.client.nodepool import Nodepool
from clarifai.utils.cli import display_co_resources, from_yaml


@cli.group(['deployment', 'dpl'])
def deployment():
  """Manage Deployments: create, delete, list"""
  pass


@deployment.command()
@click.option(
    '-np_id',
    '--nodepool_id',
    required=False,
    help='Nodepool ID for the Nodepool to interact with.')
@click.option(
    '--config',
    type=click.Path(exists=True),
    required=True,
    help='Path to the deployment config file.')
@click.option(
    '-dpl_id',
    '--deployment_id',
    required=False,
    help='New deployment ID for the deployment to create.')
@click.pass_context
def create(ctx, nodepool_id, config, deployment_id):
  """Create a new Deployment with the given config file."""
  if not nodepool_id:
    deployment_config = from_yaml(config)
    nodepool_id = deployment_config['deployment']['nodepools'][0]['id']

  nodepool = Nodepool(
      nodepool_id=nodepool_id,
      user_id=ctx.obj['user_id'],
      pat=ctx.obj['pat'],
      base_url=ctx.obj['base_url'])
  if deployment_id:
    nodepool.create_deployment(config, deployment_id=deployment_id)
  else:
    nodepool.create_deployment(config)


@deployment.command()
@click.option(
    '-np_id',
    '--nodepool_id',
    required=True,
    help='Nodepool ID for the Nodepool to interact with.')
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=16)
@click.pass_context
def list(ctx, nodepool_id, page_no, per_page):
  """List all deployments for the nodepool."""

  nodepool = Nodepool(
      nodepool_id=nodepool_id,
      user_id=ctx.obj['user_id'],
      pat=ctx.obj['pat'],
      base_url=ctx.obj['base_url'])
  response = nodepool.list_deployments(page_no=page_no, per_page=per_page)
  display_co_resources(response, "Deployment")


@deployment.command()
@click.option(
    '-np_id',
    '--nodepool_id',
    required=True,
    help='Nodepool ID for the Nodepool to interact with.')
@click.option('-dpl_id', '--deployment_id', help='Deployment ID of the nodepool to delete.')
@click.pass_context
def delete(ctx, nodepool_id, deployment_id):
  """Deletes a deployment for the nodepool."""

  nodepool = Nodepool(
      nodepool_id=nodepool_id,
      user_id=ctx.obj['user_id'],
      pat=ctx.obj['pat'],
      base_url=ctx.obj['base_url'])
  nodepool.delete_deployments([deployment_id])
