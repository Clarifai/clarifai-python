import click
from clarifai.cli.base import cli
from clarifai.client.user import User
from clarifai.utils.cli import display_co_resources


@cli.group()
@click.pass_context
def computecluster(ctx):
  """Manage Compute Clusters: create, delete, list"""
  ctx.obj = User(user_id=ctx.obj['user_id'], pat=ctx.obj['pat'], base_url=ctx.obj['base_url'])


@computecluster.command()
@click.option(
    '--compute_cluster_id',
    required=True,
    help='Compute Cluster ID for the compute cluster to create.')
@click.option(
    '--config_filepath',
    type=click.Path(exists=True),
    required=True,
    help='Path to the compute cluster config file.')
@click.pass_obj
def create(obj, compute_cluster_id, config_filepath):
  """Create a new Compute Cluster with the given id"""
  obj.create_compute_cluster(compute_cluster_id, config_filepath)


@computecluster.command()
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=16)
@click.pass_obj
def list(obj, page_no, per_page):
  """List all compute clusters for the user."""
  response = obj.list_compute_clusters(page_no, per_page)
  display_co_resources(response, "Compute Cluster")


@computecluster.command()
@click.option(
    '--compute_cluster_ids', multiple=True, help='Compute Cluster IDs of the user to delete.')
@click.pass_obj
def delete(obj, compute_cluster_ids):
  """Deletes a list of compute clusters for the user."""
  compute_cluster_ids = [compute_cluster_id for compute_cluster_id in compute_cluster_ids]
  obj.delete_compute_clusters(compute_cluster_ids)
