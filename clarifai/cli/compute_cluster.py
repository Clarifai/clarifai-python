import click

from clarifai.cli.base import cli
from clarifai.client.user import User
from clarifai.utils.cli import AliasedGroup, display_co_resources, validate_context


@cli.group(['computecluster', 'cc'], cls=AliasedGroup)
def computecluster():
  """Manage Compute Clusters: create, delete, list"""


@computecluster.command(['c'])
@click.argument('compute_cluster_id')
@click.option(
    '--config',
    type=click.Path(exists=True),
    required=True,
    help='Path to the compute cluster config file.')
@click.pass_context
def create(ctx, compute_cluster_id, config):
  """Create a new Compute Cluster with the given config file."""
  validate_context(ctx)
  user = User(
      user_id=ctx.obj.current.user_id, pat=ctx.obj.current.pat, base_url=ctx.obj.current.api_base)
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
      user_id=ctx.obj.current.user_id, pat=ctx.obj.current.pat, base_url=ctx.obj.current.api_base)
  response = user.list_compute_clusters(page_no, per_page)
  display_co_resources(
      response,
      custom_columns={
          'ID': lambda c: c.id,
          'USER_ID': lambda c: c.user_id,
          'DESCRIPTION': lambda c: c.description,
      })


@computecluster.command(['rm'])
@click.argument('compute_cluster_id')
@click.pass_context
def delete(ctx, compute_cluster_id):
  """Deletes a compute cluster for the user."""
  validate_context(ctx)
  user = User(
      user_id=ctx.obj.current.user_id, pat=ctx.obj.current.pat, base_url=ctx.obj.current.api_base)
  user.delete_compute_clusters([compute_cluster_id])
