import click
from clarifai.cli.base import cli
from clarifai.client.nodepool import Nodepool
from clarifai.utils.cli import display_co_resources


@cli.group()
@click.option(
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
@click.option('--deployment_id', required=True, help='Deployment ID for the deployment to create.')
@click.option(
    '--config_filepath',
    type=click.Path(exists=True),
    required=True,
    help='Path to the deployment config file.')
@click.pass_obj
def create(obj, deployment_id, config_filepath):
  """Create a new Deployment with the given id"""
  obj.create_deployment(deployment_id, config_filepath)


@deployment.command()
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=16)
@click.pass_obj
def list(obj, page_no, per_page):
  """List all deployments for the nodepool."""
  response = obj.list_deployments(page_no=page_no, per_page=per_page)
  display_co_resources(response, "Deployment")


@deployment.command()
@click.option('--deployment_ids', multiple=True, help='Deployment IDs of the nodepool to delete.')
@click.pass_obj
def delete(obj, deployment_ids):
  """Deletes a list of deployments for the nodepool."""
  deployment_ids = [deployment_id for deployment_id in deployment_ids]
  obj.delete_deployments(deployment_ids)
