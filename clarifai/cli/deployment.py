import click
from clarifai.cli.base import cli
from clarifai.client.nodepool import Nodepool
from clarifai.utils.cli import display_co_resources


@cli.group()
@click.option(
    '-np_id',
    '--nodepool_id', required=True, help='Nodepool ID for the Nodepool to interact with.')
@click.pass_context
def deployment(ctx, nodepool_id):
  """Manage Deployments: create, delete, list"""
  ctx.obj = Nodepool(
      nodepool_id=nodepool_id,
      user_id=ctx.obj['user_id'],
      pat=ctx.obj['pat'],
      base_url=ctx.obj['base_url'])


@deployment.command()
@click.option(
    '-config',
    '--config_filepath',
    type=click.Path(exists=True),
    required=True,
    help='Path to the deployment config file.')
@click.option('-dpl_id', '--deployment_id', required=False, help='New deployment ID for the deployment to create.')
@click.pass_obj
def create(obj, config_filepath, deployment_id):
  """Create a new Deployment with the given config file."""
  if deployment_id:
    obj.create_deployment(config_filepath, deployment_id=deployment_id)
  else:
    obj.create_deployment(config_filepath)


@deployment.command()
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=16)
@click.pass_obj
def list(obj, page_no, per_page):
  """List all deployments for the nodepool."""
  response = obj.list_deployments(page_no=page_no, per_page=per_page)
  display_co_resources(response, "Deployment")


@deployment.command()
@click.option('-dpl_id', '--deployment_id', help='Deployment ID of the nodepool to delete.')
@click.pass_obj
def delete(obj, deployment_id):
  """Deletes a deployment for the nodepool."""
  obj.delete_deployments([deployment_id])
