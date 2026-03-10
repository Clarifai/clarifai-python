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
    """Manage deployments."""


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

    # Convert generators to list so we can check length
    response = [item for item in response]

    if not response:
        click.echo("\nNo deployments found.")
        click.echo("  Deploy a model: clarifai model deploy ./my-model --instance gpu-nvidia-a10g")
        return

    def _status_label(dep):
        return "Enabled" if dep.status == 0 else "Disabled"

    def _replicas(dep):
        m = dep.deployment_metrics
        if m:
            return f"{m.live_replicas}/{m.desired_replicas}"
        return "-"

    display_co_resources(
        response,
        custom_columns={
            'ID': lambda c: c.id,
            'STATUS': _status_label,
            'REPLICAS': _replicas,
            'MODEL_ID': lambda c: c.worker.model.id,
            'MODEL_VERSION': lambda c: c.worker.model.model_version.id[:12]
            if c.worker.model.model_version.id
            else "-",
            'NODEPOOL_ID': lambda c: c.nodepools[0].id if c.nodepools else "-",
            'COMPUTE_CLUSTER': lambda c: c.nodepools[0].compute_cluster.id
            if c.nodepools and c.nodepools[0].compute_cluster
            else "-",
        },
    )


@deployment.command(['get', 'status'])
@click.argument('deployment_id')
@click.pass_context
def get(ctx, deployment_id):
    """Show details for a single deployment.

    \b
    Shows status (enabled/disabled), live/desired replicas, rollout state,
    instance type with GPU info, nodepool, compute cluster, and timing.

    \b
    Examples:
      clarifai deployment get deploy-abc123
      clarifai deployment status deploy-abc123
    """
    validate_context(ctx)

    from clarifai.cli.model import _print_deployment_detail  # noqa: E402
    from clarifai.errors import UserError
    from clarifai.runners.models.model_deploy import get_deployment

    try:
        dep = get_deployment(
            deployment_id,
            user_id=ctx.obj.current.user_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
        )
        _print_deployment_detail(dep)
    except UserError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"), err=True)
        raise SystemExit(1)


@deployment.command()
@click.argument('deployment_id')
@click.option(
    '--follow/--no-follow',
    default=True,
    help='Continuously tail new logs. Use --no-follow to print and exit.',
)
@click.option(
    '--duration',
    default=None,
    type=int,
    help='Stop after N seconds (default: unlimited, Ctrl+C to stop).',
)
@click.option(
    '--log-type',
    default='model',
    type=click.Choice(['model', 'events'], case_sensitive=False),
    help='Log type: model (stdout/stderr) or events (k8s scheduling/scaling).',
)
@click.pass_context
def logs(ctx, deployment_id, follow, duration, log_type):
    """Stream logs from a deployment's runner.

    \b
    Resolves the model, version, and nodepool from the deployment
    and streams runner stdout/stderr or k8s events.

    \b
    Examples:
      clarifai deployment logs deploy-abc123
      clarifai deployment logs deploy-abc123 --log-type events
      clarifai deployment logs deploy-abc123 --no-follow
      clarifai deployment logs deploy-abc123 --duration 60
    """
    validate_context(ctx)

    from clarifai.errors import UserError
    from clarifai.runners.models.model_deploy import get_deployment, stream_model_logs

    user_id = ctx.obj.current.user_id
    pat = ctx.obj.current.pat
    base_url = ctx.obj.current.api_base

    try:
        dep = get_deployment(deployment_id, user_id=user_id, pat=pat, base_url=base_url)
    except UserError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"), err=True)
        raise SystemExit(1)

    # Extract model/version/nodepool from deployment proto
    model_id = app_id = model_version_id = None
    compute_cluster_id = nodepool_id = None

    w = dep.worker
    if w and w.model:
        model_id = w.model.id
        user_id = w.model.user_id or user_id
        app_id = w.model.app_id
        if w.model.model_version and w.model.model_version.id:
            model_version_id = w.model.model_version.id
    if dep.nodepools:
        np = dep.nodepools[0]
        nodepool_id = np.id
        if np.compute_cluster and np.compute_cluster.id:
            compute_cluster_id = np.compute_cluster.id

    # Map user-friendly names to API log_type values
    api_log_type = {"model": "runner", "events": "runner.events"}[log_type.lower()]

    try:
        stream_model_logs(
            model_id=model_id,
            user_id=user_id,
            app_id=app_id,
            model_version_id=model_version_id,
            compute_cluster_id=compute_cluster_id,
            nodepool_id=nodepool_id,
            pat=pat,
            base_url=base_url,
            follow=follow,
            duration=duration,
            log_type=api_log_type,
        )
    except UserError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"), err=True)
        raise SystemExit(1)


@deployment.command(['rm'])
@click.argument('deployment_id')
@click.pass_context
def delete(ctx, deployment_id):
    """Delete a deployment.

    \b
    Examples:
      clarifai deployment rm deploy-abc123
    """
    validate_context(ctx)

    from clarifai.errors import UserError
    from clarifai.runners.models.model_deploy import delete_deployment

    try:
        delete_deployment(
            deployment_id,
            user_id=ctx.obj.current.user_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
        )
        click.echo(click.style(f"  Deployment '{deployment_id}' deleted.", fg="green"))
    except UserError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"), err=True)
        raise SystemExit(1)
