import shutil

import click

from clarifai.cli.base import cli
from clarifai.utils.cli import AliasedGroup, validate_context


@cli.group(
    ['app', 'a'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def app():
    """Manage Apps: create, delete, list"""


@app.command(['c'])
@click.argument('app_id')
@click.option(
    '--base-workflow',
    default='Empty',
    help='Base workflow to use for the app. Examples: Universal, Language-Understanding, General',
)
@click.pass_context
def create(ctx, app_id, base_workflow):
    """Create a new App with the given app ID."""
    from clarifai.client.user import User

    validate_context(ctx)
    user = User(
        user_id=ctx.obj.current.user_id, pat=ctx.obj.current.pat, base_url=ctx.obj.current.api_base
    )
    user.create_app(app_id=app_id, base_workflow=base_workflow)
    click.echo(f"App '{app_id}' created successfully.")


@app.command(['ls'])
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=16)
@click.option(
    '--user_id',
    required=False,
    help='User ID to list apps for (defaults to current user).',
    default=None,
)
@click.pass_context
def list(ctx, page_no, per_page, user_id):
    """List all apps for the user."""
    from clarifai.client.user import User

    validate_context(ctx)
    # Use provided user_id or fall back to current context's user_id
    target_user_id = user_id if user_id else ctx.obj.current.user_id
    user = User(user_id=target_user_id, pat=ctx.obj.current.pat, base_url=ctx.obj.current.api_base)
    apps = [app for app in user.list_apps(page_no=page_no, per_page=per_page)]

    if not apps:
        click.echo("No apps found.")
        return

    # Display apps in a simple table
    from tabulate import tabulate

    rows = []
    for app in apps:
        rows.append([app.user_id, app.id])

    table = tabulate(rows, headers=["User ID", "App ID"], tablefmt="plain")
    click.echo(table)


@app.command(['rm'])
@click.argument('app_id')
@click.pass_context
def delete(ctx, app_id):
    """Deletes an app for the user."""
    from clarifai.client.user import User

    validate_context(ctx)
    user = User(
        user_id=ctx.obj.current.user_id, pat=ctx.obj.current.pat, base_url=ctx.obj.current.api_base
    )
    user.delete_app(app_id)
    click.echo(f"App '{app_id}' deleted successfully.")
