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
    '-config',
    '--config_filepath',
    type=click.Path(exists=True),
    required=True,
    help='Path to the compute cluster config file.')
@click.option(
    '-cc_id',
    '--compute_cluster_id',
    required=False,
    help='New Compute Cluster ID for the compute cluster to create.')
@click.pass_obj
def create(obj, config_filepath, compute_cluster_id):
  """Create a new Compute Cluster with the given config file."""
  if compute_cluster_id:
    obj.create_compute_cluster(config_filepath, compute_cluster_id=compute_cluster_id)
  else:
    obj.create_compute_cluster(config_filepath)


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
    '-cc_id',
    '--compute_cluster_id', help='Compute Cluster ID of the user to delete.')
@click.pass_obj
def delete(obj, compute_cluster_id):
  """Deletes a compute cluster for the user."""
  obj.delete_compute_clusters([compute_cluster_id])
