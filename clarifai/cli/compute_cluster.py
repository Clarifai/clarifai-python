import click

from clarifai.cli.base import cli
from clarifai.client.user import User
from clarifai.utils.cli import display_co_resources, AliasedGroup, validate_context


@cli.group(['computecluster', 'cc'], cls=AliasedGroup)
def computecluster():
  """Manage Compute Clusters: create, delete, list"""


@computecluster.command(['c'])
@click.option(
    '--config',
    type=click.Path(exists=True),
    required=True,
    help='Path to the compute cluster config file.')
@click.option(
    '-cc_id',
    '--compute_cluster_id',
    required=False,
    help='New Compute Cluster ID for the compute cluster to create.')
@click.pass_context
def create(ctx, config, compute_cluster_id):
  """Create a new Compute Cluster with the given config file."""
  validate_context(ctx)
  user = User(
      user_id=ctx.obj.contexts[ctx.obj.current_context].user_id,
      pat=ctx.obj.contexts[ctx.obj.current_context].pat,
      base_url=ctx.obj.contexts[ctx.obj.current_context].base_url)
  if compute_cluster_id:
    user.create_compute_cluster(config, compute_cluster_id=compute_cluster_id)
  else:
    user.create_compute_cluster(config)


@computecluster.command(['ls'])
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=16)
@click.pass_context
def list(ctx, page_no, per_page):
  """List all compute clusters for the user."""
  validate_context(ctx)
  user = User(
      user_id=ctx.obj.contexts[ctx.obj.current_context].user_id,
      pat=ctx.obj.contexts[ctx.obj.current_context].pat,
      base_url=ctx.obj.contexts[ctx.obj.current_context].base_url)
  response = user.list_compute_clusters(page_no, per_page)
  display_co_resources(response, "Compute Cluster")


@computecluster.command(['rm'])
@click.option('-cc_id', '--compute_cluster_id', help='Compute Cluster ID of the user to delete.')
@click.pass_context
def delete(ctx, compute_cluster_id):
  """Deletes a compute cluster for the user."""
  validate_context(ctx)
  user = User(
      user_id=ctx.obj.contexts[ctx.obj.current_context].user_id,
      pat=ctx.obj.contexts[ctx.obj.current_context].pat,
      base_url=ctx.obj.contexts[ctx.obj.current_context].base_url)
  user.delete_compute_clusters([compute_cluster_id])
