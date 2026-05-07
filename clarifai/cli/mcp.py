"""MCP CLI: merchandised shortcuts over the model engine.

Wraps `clarifai model init --toolkit mcp`, `clarifai model deploy`, and
list-models so MCP tool builders get an opinionated, friendly surface
without losing access to the general-purpose `model` commands.
"""

import os
import shutil

import click

from clarifai.cli.base import cli
from clarifai.errors import UserError
from clarifai.utils.cli import (
    AliasedGroup,
    from_yaml,
    validate_context,
)
from clarifai.utils.logging import logger


@cli.group(
    ['mcp'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def mcp():
    """Build, deploy, and manage MCP (Model Context Protocol) servers.

    \b
    Workflow:   init → (model serve) → deploy → list

    \b
    MCP servers are models under the hood. These commands are opinionated
    shortcuts. For everything else, use the full model engine:

    \b
      clarifai model serve      # local testing
      clarifai model upload     # upload without deploying
      clarifai model logs       # stream deployment logs
      clarifai model status     # check deployment health
      clarifai model undeploy   # tear down a deployment
      clarifai model list       # list all models (not just MCP)
    """


@mcp.command()
@click.argument("model_path", type=click.Path(), required=False, default=None)
@click.pass_context
def init(ctx, model_path):
    """Scaffold a new MCP server project.

    \b
    Creates a ready-to-use MCP server directory with everything needed
    to define tools, resources, and prompts — then serve or deploy them.

    \b
    MODEL_PATH  Project directory name or path (default: current directory).

    \b
    Examples:
      clarifai mcp init my-search-tool
      clarifai mcp init                  # scaffold in current directory

    \b
    Next steps:
      clarifai model serve ./my-search-tool   # test locally
      clarifai mcp deploy ./my-search-tool    # deploy to Clarifai
    """
    # Defer import to avoid cycles and to honor the lazy-loading pattern.
    from clarifai.cli.model import init as model_init

    ctx.invoke(
        model_init,
        model_path=model_path,
        toolkit='mcp',
        model_name=None,
        streaming_video=False,
    )

    _apply_mcp_compute_defaults(model_path)


def _apply_mcp_compute_defaults(model_path):
    """Override the template's GPU default with a CPU instance for MCP projects.

    The shared model-init template hard-codes `compute.instance: g5.xlarge`,
    which is wrong for MCP servers. MCP runs on CPU (see compute_presets.py).
    """
    target_dir = os.path.abspath(model_path) if model_path else os.getcwd()
    config_path = os.path.join(target_dir, 'config.yaml')
    if not os.path.isfile(config_path):
        return

    try:
        import yaml

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}

        if (cfg.get('model') or {}).get('model_type_id') != 'mcp':
            return

        compute = cfg.setdefault('compute', {})
        # Only override the template's GPU default; respect any explicit user choice.
        if compute.get('instance') in (None, '', 'g5.xlarge'):
            compute['instance'] = 't3a.2xlarge'
            with open(config_path, 'w') as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Could not apply MCP compute defaults: {exc}")


def _read_mcp_config(model_path):
    """Read config.yaml and verify it's an MCP project.

    Returns the parsed config dict on success. Raises UserError otherwise.
    """
    config_path = os.path.join(model_path, 'config.yaml')
    if not os.path.isfile(config_path):
        raise UserError(
            f"No config.yaml found at '{config_path}'.\n"
            "  Initialize an MCP project first:\n"
            "    clarifai mcp init"
        )
    config = from_yaml(config_path)
    model_type_id = config.get('model', {}).get('model_type_id')
    if model_type_id != 'mcp':
        raise UserError(
            f"'{model_path}' is not an MCP project (model_type_id='{model_type_id}').\n"
            "  Use 'clarifai model deploy' for non-MCP models, or scaffold an MCP\n"
            "  project with 'clarifai mcp init'."
        )
    return config


def _make_url_helper(base_url, pat):
    """Construct a ClarifaiUrlHelper bound to the active context's base/PAT."""
    from clarifai.client.auth.helper import ClarifaiAuthHelper
    from clarifai.urls.helper import ClarifaiUrlHelper

    auth = ClarifaiAuthHelper(
        user_id="",
        app_id="",
        pat=pat or "",
        base=base_url or "https://api.clarifai.com",
        validate=False,
    )
    return ClarifaiUrlHelper(auth=auth)


def _print_mcp_endpoint(result, base_url, pat):
    """Append an MCP-specific footer (endpoint URL + connect hint) to deploy output."""
    from clarifai.runners.models import deploy_output as out

    helper = _make_url_helper(base_url, pat)
    endpoint = helper.mcp_api_url(
        user_id=result['user_id'],
        app_id=result['app_id'],
        model_id=result['model_id'],
    )

    click.echo()
    out.phase_header("MCP Endpoint")
    out.link("URL", endpoint)
    click.echo()
    click.echo(click.style("  Connect an MCP client:", bold=True))
    click.echo(f"    URL:    {endpoint}")
    click.echo("    Header: Authorization: Bearer $CLARIFAI_PAT")


@mcp.command()
@click.argument('model_path', type=click.Path(), required=False, default=None)
@click.option(
    '--instance',
    default=None,
    help='Hardware instance type. Auto-selects a CPU instance if omitted (MCP servers rarely need GPU).',
)
@click.option(
    '--model-url',
    default=None,
    help='Deploy an already-uploaded MCP model by its Clarifai URL (skips upload).',
)
@click.option(
    '--model-version-id',
    default=None,
    help='Specific model version to deploy (default: latest).',
)
@click.option(
    '--min-replicas',
    default=1,
    type=int,
    show_default=True,
    help='Minimum number of running replicas.',
)
@click.option(
    '--max-replicas',
    default=5,
    type=int,
    show_default=True,
    help='Maximum replicas for autoscaling.',
)
@click.option(
    '--cloud',
    default=None,
    help='Cloud provider (e.g., aws, gcp). Auto-detected from --instance if omitted.',
)
@click.option(
    '--region',
    default=None,
    help='Cloud region (e.g., us-east-1). Auto-detected from --instance if omitted.',
)
@click.option(
    '--compute-cluster-id',
    default=None,
    help='[Advanced] Existing compute cluster ID (skip auto-creation).',
)
@click.option(
    '--nodepool-id',
    default=None,
    help='[Advanced] Existing nodepool ID (skip auto-creation).',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Show detailed build, upload, and deployment logs.',
)
@click.pass_context
def deploy(
    ctx,
    model_path,
    instance,
    model_url,
    model_version_id,
    min_replicas,
    max_replicas,
    cloud,
    region,
    compute_cluster_id,
    nodepool_id,
    verbose,
):
    """Deploy an MCP server to Clarifai cloud compute.

    \b
    Wraps `clarifai model deploy` with MCP-aware defaults: a CPU instance
    is auto-selected when --instance is omitted, and the MCP endpoint URL
    is printed alongside the standard deploy output.

    \b
    MODEL_PATH  Local MCP project directory (default: ".").
                Not needed when using --model-url.

    \b
    Examples:
      clarifai mcp deploy ./my-search-tool
      clarifai mcp deploy --model-url https://clarifai.com/user/app/models/my-mcp
      clarifai mcp deploy ./my-search-tool --instance t3a.2xlarge
    """
    validate_context(ctx)

    if model_path and model_url:
        raise UserError("Specify only one of: MODEL_PATH or --model-url.")

    # Default to current directory when neither is given (matches model deploy)
    resolved_path = None
    if not model_url:
        resolved_path = os.path.abspath(model_path or ".")
        if not os.path.isdir(resolved_path):
            raise click.BadParameter(f"Model path '{resolved_path}' is not a directory.")
        # Pre-flight: confirm it's an MCP project (gives a friendly error early,
        # before the model is uploaded).
        _read_mcp_config(resolved_path)

    from clarifai.cli.model import _print_deploy_result
    from clarifai.runners.models.model_deploy import ModelDeployer

    user_id = ctx.obj.current.user_id
    app_id = getattr(ctx.obj.current, 'app_id', None)
    pat = ctx.obj.current.pat
    base_url = ctx.obj.current.api_base

    deployer = ModelDeployer(
        model_path=resolved_path,
        model_url=model_url,
        user_id=user_id,
        app_id=app_id,
        model_version_id=model_version_id,
        instance_type=instance,
        cloud_provider=cloud,
        region=region,
        compute_cluster_id=compute_cluster_id,
        nodepool_id=nodepool_id,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        pat=pat,
        base_url=base_url,
        verbose=verbose,
    )

    result = deployer.deploy()
    _print_deploy_result(result)
    _print_mcp_endpoint(result, base_url=base_url, pat=pat)


@mcp.command(name="list")
@click.option(
    '-u',
    '--user-id',
    type=str,
    default=None,
    help='User ID to list MCP servers for (default: current user). Use "all" for the public catalog.',
)
@click.option(
    '-a',
    '--app-id',
    type=str,
    default=None,
    help='Filter by app ID.',
)
@click.pass_context
def list_mcp(ctx, user_id, app_id):
    """List MCP servers (filtered view of model deployments).

    \b
    Shows MCP models you've created along with their MCP endpoint URLs.
    For deployment status, run 'clarifai model status <user>/<app>/models/<id>'.

    \b
    Examples:
      clarifai mcp list
      clarifai mcp list --app-id my-app
      clarifai mcp list --user-id all     # public MCP servers
    """
    validate_context(ctx)

    from tabulate import tabulate

    from clarifai.client.user import User

    pat = ctx.obj.current.pat
    base_url = ctx.obj.current.api_base
    effective_user_id = user_id or ctx.obj.current.user_id

    user = User(user_id=effective_user_id, pat=pat, base_url=base_url)
    all_models = user.list_models(
        user_id=effective_user_id, app_id=app_id, show=False, return_clarifai_model=False
    )

    helper = _make_url_helper(base_url, pat)

    rows = []
    for m in all_models:
        if m.get('model_type') != 'mcp':
            continue
        endpoint = helper.mcp_api_url(user_id=m['user_id'], app_id=m['app_id'], model_id=m['id'])
        rows.append(
            {
                'NAME': m['id'],
                'APP': f"{m['user_id']}/{m['app_id']}",
                'MCP ENDPOINT': endpoint,
            }
        )

    if not rows:
        click.echo("No MCP servers found.")
        click.echo()
        click.echo("  Create one with:")
        click.echo("    clarifai mcp init my-mcp-server")
        click.echo("    clarifai mcp deploy ./my-mcp-server")
        return

    click.echo(tabulate(rows, headers="keys"))
    click.echo()
    click.echo(
        f"  {len(rows)} MCP server(s). Use 'clarifai model status <user>/<app>/models/<id>' for deployment details."
    )

    # Reference logger so unused-import lints stay clean if we later add diagnostics.
    _ = logger
