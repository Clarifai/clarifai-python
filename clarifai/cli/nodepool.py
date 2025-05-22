import shutil

import click

from clarifai.cli.base import cli
from clarifai.utils.cli import (
    AliasedGroup,
    display_co_resources,
    dump_yaml,
    from_yaml,
    validate_context,
)


@cli.group(
    ['nodepool', 'np'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def nodepool():
    """Manage Nodepools: create, delete, list"""


@nodepool.command(['c'])
@click.argument('compute_cluster_id')
@click.argument('nodepool_id')
@click.option(
    '--config',
    type=click.Path(exists=True),
    required=True,
    help='Path to the nodepool config file.',
)
@click.pass_context
def create(ctx, compute_cluster_id, nodepool_id, config):
    """Create a new Nodepool with the given config file."""
    from clarifai.client.compute_cluster import ComputeCluster

    validate_context(ctx)
    nodepool_config = from_yaml(config)
    if not compute_cluster_id:
        if 'compute_cluster' not in nodepool_config['nodepool']:
            click.echo(
                "Please provide a compute cluster ID either in the config file or using --compute_cluster_id flag",
                err=True,
            )
            return
        compute_cluster_id = nodepool_config['nodepool']['compute_cluster']['id']
    elif 'compute_cluster' not in nodepool_config['nodepool']:
        nodepool_config['nodepool']['compute_cluster']['id'] = compute_cluster_id
        dump_yaml(config, nodepool_config)

    compute_cluster = ComputeCluster(
        compute_cluster_id=compute_cluster_id,
        user_id=ctx.obj.current.user_id,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
    )
    if nodepool_id:
        compute_cluster.create_nodepool(config, nodepool_id=nodepool_id)
    else:
        compute_cluster.create_nodepool(config)


@nodepool.command(['ls'])
@click.argument('compute_cluster_id', default="")
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=128)
@click.pass_context
def list(ctx, compute_cluster_id, page_no, per_page):
    """List all nodepools for the user across all compute clusters. If compute_cluster_id is provided
    it will list only within that compute cluster."""
    from clarifai.client.compute_cluster import ComputeCluster
    from clarifai.client.user import User

    validate_context(ctx)

    cc_id = compute_cluster_id

    if cc_id:
        compute_cluster = ComputeCluster(
            compute_cluster_id=cc_id,
            user_id=ctx.obj.current.user_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
        )
        response = compute_cluster.list_nodepools(page_no, per_page)
    else:
        user = User(
            user_id=ctx.obj.current.user_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
        )
        ccs = user.list_compute_clusters(page_no, per_page)
        response = []
        for cc in ccs:
            compute_cluster = ComputeCluster(
                compute_cluster_id=cc.id,
                user_id=ctx.obj.current.user_id,
                pat=ctx.obj.current.pat,
                base_url=ctx.obj.current.api_base,
            )
            response.extend([i for i in compute_cluster.list_nodepools(page_no, per_page)])

    display_co_resources(
        response,
        custom_columns={
            'ID': lambda c: c.id,
            'USER_ID': lambda c: c.compute_cluster.user_id,
            'COMPUTE_CLUSTER_ID': lambda c: c.compute_cluster.id,
            'DESCRIPTION': lambda c: c.description,
        },
    )


@nodepool.command(['rm'])
@click.argument('compute_cluster_id')
@click.argument('nodepool_id')
@click.pass_context
def delete(ctx, compute_cluster_id, nodepool_id):
    """Deletes a nodepool for the user."""
    from clarifai.client.compute_cluster import ComputeCluster

    validate_context(ctx)
    compute_cluster = ComputeCluster(
        compute_cluster_id=compute_cluster_id,
        user_id=ctx.obj.current.user_id,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
    )
    compute_cluster.delete_nodepools([nodepool_id])
