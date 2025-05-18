import shutil

import click

from clarifai.cli.base import cli
from clarifai.utils.cli import AliasedGroup, display_co_resources, from_yaml, validate_context


@cli.group(
    ['deployment', 'dp'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def deployment():
    """Manage Deployments: create, delete, list"""


@deployment.command(['c'])
@click.argument('nodepool_id')
@click.argument('deployment_id')
@click.option(
    '--config',
    type=click.Path(exists=True),
    required=True,
    help='Path to the deployment config file.',
)
@click.pass_context
def create(ctx, nodepool_id, deployment_id, config):
    """Create a new Deployment with the given config file."""

    from clarifai.client.nodepool import Nodepool

    validate_context(ctx)
    if not nodepool_id:
        deployment_config = from_yaml(config)
        nodepool_id = deployment_config['deployment']['nodepools'][0]['id']

    nodepool = Nodepool(
        nodepool_id=nodepool_id,
        user_id=ctx.obj.current.user_id,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
    )
    if deployment_id:
        nodepool.create_deployment(config, deployment_id=deployment_id)
    else:
        nodepool.create_deployment(config)


@deployment.command(['ls'])
@click.argument('nodepool_id', default="")
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=16)
@click.pass_context
def list(ctx, nodepool_id, page_no, per_page):
    """List all deployments for the nodepool."""
    from clarifai.client.compute_cluster import ComputeCluster
    from clarifai.client.nodepool import Nodepool
    from clarifai.client.user import User

    validate_context(ctx)
    if nodepool_id:
        nodepool = Nodepool(
            nodepool_id=nodepool_id,
            user_id=ctx.obj.current.user_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
        )
        response = nodepool.list_deployments(page_no=page_no, per_page=per_page)
    else:
        user = User(
            user_id=ctx.obj.current.user_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
        )
        ccs = user.list_compute_clusters(page_no, per_page)
        nps = []
        for cc in ccs:
            compute_cluster = ComputeCluster(
                compute_cluster_id=cc.id,
                user_id=ctx.obj.current.user_id,
                pat=ctx.obj.current.pat,
                base_url=ctx.obj.current.api_base,
            )
            nps.extend([i for i in compute_cluster.list_nodepools(page_no, per_page)])
        response = []
        for np in nps:
            nodepool = Nodepool(
                nodepool_id=np.id,
                user_id=ctx.obj.current.user_id,
                pat=ctx.obj.current.pat,
                base_url=ctx.obj.current.api_base,
            )
            response.extend(
                [i for i in nodepool.list_deployments(page_no=page_no, per_page=per_page)]
            )

    display_co_resources(
        response,
        custom_columns={
            'ID': lambda c: c.id,
            'USER_ID': lambda c: c.user_id,
            'COMPUTE_CLUSTER_ID': lambda c: c.nodepools[0].compute_cluster.id,
            'NODEPOOL_ID': lambda c: c.nodepools[0].id,
            'MODEL_USER_ID': lambda c: c.worker.model.user_id,
            'MODEL_APP_ID': lambda c: c.worker.model.app_id,
            'MODEL_ID': lambda c: c.worker.model.id,
            'MODEL_VERSION_ID': lambda c: c.worker.model.model_version.id,
            'DESCRIPTION': lambda c: c.description,
        },
    )


@deployment.command(['rm'])
@click.argument('nodepool_id')
@click.argument('deployment_id')
@click.pass_context
def delete(ctx, nodepool_id, deployment_id):
    """Deletes a deployment for the nodepool."""
    from clarifai.client.nodepool import Nodepool

    validate_context(ctx)
    nodepool = Nodepool(
        nodepool_id=nodepool_id,
        user_id=ctx.obj.current.user_id,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
    )
    nodepool.delete_deployments([deployment_id])
