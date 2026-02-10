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
@click.option(
    '--config',
    type=click.Path(exists=True),
    required=True,
    help='Path to the deployment config YAML file.',
)
@click.pass_context
def create(ctx, nodepool_id, deployment_id, config):
    """
    Create a new Deployment from a config file.

    The config file is a YAML that defines the worker (model or workflow),
    nodepools, autoscale settings, and visibility.

    Ex: clarifai deployment create --config deployment.yaml

    Example deployment.yaml:

    \b
    deployment:
      id: "my-deployment"
      worker:
        model:
          id: "model-id"
          model_version:
            id: "version-id"
          user_id: "owner-id"
          app_id: "app-id"
      nodepools:
        - id: "nodepool-id"
          compute_cluster:
            id: "cluster-id"
            user_id: "cluster-owner-id"
      autoscale_config:
        min_replicas: 1
        max_replicas: 1
        scale_to_zero_delay_seconds: 300
      deploy_latest_version: true

    """

    from clarifai.client.nodepool import Nodepool

    validate_context(ctx)
    deployment_config = from_yaml(config)
    nodepool_id = deployment_config['deployment']['nodepools'][0]['id']
    deployment_id = deployment_config['deployment']['id']

    nodepool = Nodepool(
        nodepool_id=nodepool_id,
        user_id=ctx.obj.current.user_id,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
    )
    nodepool.create_deployment(config, deployment_id=deployment_id)


@deployment.command(['ls'])
@click.option('--nodepool_id', required=False, help='Nodepool ID to list deployments for.')
@click.option(
    '--compute_cluster_id', required=False, help='Compute cluster ID to list deployments for.'
)
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=16)
@click.pass_context
def list(ctx, nodepool_id, compute_cluster_id, page_no, per_page):
    """List all deployments for the nodepool."""
    from clarifai_grpc.grpc.api import resources_pb2

    from clarifai.client.compute_cluster import ComputeCluster
    from clarifai.client.nodepool import Nodepool
    from clarifai.client.user import User

    validate_context(ctx)
    if nodepool_id:
        kwargs = {}
        if compute_cluster_id:
            kwargs['compute_cluster'] = resources_pb2.ComputeCluster(id=compute_cluster_id)
        nodepool = Nodepool(
            nodepool_id=nodepool_id,
            user_id=ctx.obj.current.user_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
            **kwargs,
        )
        response = nodepool.list_deployments(page_no=page_no, per_page=per_page)
    else:
        if compute_cluster_id:
            ccs = [
                ComputeCluster(
                    compute_cluster_id=compute_cluster_id,
                    user_id=ctx.obj.current.user_id,
                    pat=ctx.obj.current.pat,
                    base_url=ctx.obj.current.api_base,
                )
            ]
        else:
            user = User(
                user_id=ctx.obj.current.user_id,
                pat=ctx.obj.current.pat,
                base_url=ctx.obj.current.api_base,
            )
            all_ccs = user.list_compute_clusters(page_no, per_page)
            ccs = [
                ComputeCluster(
                    compute_cluster_id=cc.id,
                    user_id=ctx.obj.current.user_id,
                    pat=ctx.obj.current.pat,
                    base_url=ctx.obj.current.api_base,
                )
                for cc in all_ccs
            ]

        nps = []
        for compute_cluster in ccs:
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
