import click
from clarifai.cli.base import cli
from clarifai.client.compute_cluster import ComputeCluster
from clarifai.utils.cli import display_co_resources


@cli.group()
@click.option(
    '--compute_cluster_id',
    required=True,
    help='Compute Cluster ID for the compute cluster to interact with.')
@click.pass_context
def nodepool(ctx, compute_cluster_id):
  """Manage Nodepools: create, delete, list"""
  ctx.obj = ComputeCluster(
      compute_cluster_id=compute_cluster_id,
      user_id=ctx.obj['user_id'],
      pat=ctx.obj['pat'],
      base_url=ctx.obj['base_url'])


@nodepool.command()
@click.option('--nodepool_id', required=True, help='Nodepool ID for the nodepool to create.')
@click.option(
    '--config_filepath',
    type=click.Path(exists=True),
    required=True,
    help='Path to the nodepool config file.')
@click.pass_obj
def create(obj, nodepool_id, config_filepath):
  """Create a new Nodepool with the given id"""
  obj.create_nodepool(nodepool_id, config_filepath)


@nodepool.command()
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=16)
@click.pass_obj
def list(obj, page_no, per_page):
  """List all nodepools for the user."""
  response = obj.list_nodepools(page_no, per_page)
  display_co_resources(response, "Nodepool")


@nodepool.command()
@click.option('--nodepool_ids', multiple=True, help='Nodepool IDs of the user to delete.')
@click.pass_obj
def delete(obj, nodepool_ids):
  """Deletes a list of nodepools for the user."""
  nodepool_ids = [nodepool_id for nodepool_id in nodepool_ids]
  obj.delete_nodepools(nodepool_ids)
