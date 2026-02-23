import json
import os
import sys

import click
import yaml

from clarifai import __version__
from clarifai.utils.cli import AliasedGroup, TableFormatter, load_command_modules, masked_input
from clarifai.utils.config import Config, Context
from clarifai.utils.constants import DEFAULT_BASE, DEFAULT_CONFIG, DEFAULT_UI
from clarifai.utils.logging import logger


@click.group(cls=AliasedGroup)
@click.version_option(version=__version__)
@click.option('--config', default=DEFAULT_CONFIG)
@click.pass_context
def cli(ctx, config):
    """Clarifai CLI"""
    ctx.ensure_object(dict)
    if os.path.exists(config):
        cfg = Config.from_yaml(filename=config)
        ctx.obj = cfg
    else:
        cfg = Config(
            filename=config,
            current_context='default',
            contexts={
                'default': Context(
                    'default',
                    CLARIFAI_PAT=os.environ.get('CLARIFAI_PAT', ''),
                    CLARIFAI_USER_ID=os.environ.get('CLARIFAI_USER_ID', ''),
                    CLARIFAI_API_BASE=DEFAULT_BASE,
                    CLARIFAI_UI=DEFAULT_UI,
                )
            },
        )
        try:
            cfg.to_yaml(config)
        except Exception:
            logger.warning(
                "Could not write configuration to disk. Could be a read only file system."
            )
        ctx.obj = cfg  # still have the default config even if couldn't write.


@cli.command()
@click.argument('shell', type=click.Choice(['bash', 'zsh']))
def shell_completion(shell):
    """Shell completion script"""
    os.system(f"_CLARIFAI_COMPLETE={shell}_source clarifai")


@cli.group(cls=AliasedGroup)
def config():
    """
    Manage multiple configuration profiles (contexts).

    Authentication Precedence:\n
      1. Environment variables (e.g., `CLARIFAI_PAT`) are used first if set.
      2. The settings from the active context are used if no environment variables are provided.\n
    """


def _get_pat_interactive():
    """Resolve PAT from --pat flag, env var, or interactive prompt. Returns (pat, source)."""
    env_pat = os.environ.get('CLARIFAI_PAT')
    if env_pat:
        click.secho('Using PAT from CLARIFAI_PAT environment variable.', fg='cyan')
        return env_pat

    # Best-effort browser open to PAT page
    pat_url = 'https://clarifai.com/me/settings/secrets'
    try:
        import webbrowser

        webbrowser.open(pat_url)
        click.secho(f'Opening browser to {pat_url} ...', fg='cyan')
    except Exception:
        click.echo(f'Get your PAT at: {click.style(pat_url, fg="cyan", underline=True)}')
    return masked_input('Enter your PAT: ')


def _verify_and_resolve_user(pat, api_url):
    """Validate PAT and return the personal user_id via GET /v2/users/me."""
    try:
        from clarifai.client.user import User

        user = User(user_id='me', pat=pat, base_url=api_url)
        response = user.get_user_info(user_id='me')
        return response.user.id
    except Exception as e:
        click.secho(f'Authentication failed: {e}', fg='red', err=True)
        click.echo(
            f'Create a PAT at: {click.style("https://clarifai.com/me/settings/secrets", fg="cyan", underline=True)}',
            err=True,
        )
        raise click.Abort()


def _list_user_orgs(pat, user_id, api_url):
    """Return list of (org_id, org_name) tuples for the user. Returns [] on failure."""
    try:
        from clarifai.client.user import User

        user = User(user_id=user_id, pat=pat, base_url=api_url)
        orgs = user.list_organizations()
        return [(org['id'], org['name']) for org in orgs if org.get('id')]
    except Exception:
        return []


def _prompt_user_or_org(personal_user_id, orgs):
    """Interactive org selection. Returns selected user_id."""
    click.echo()
    choices = [(personal_user_id, '(personal)')]
    for org_id, org_name in orgs:
        label = f'({org_name})' if org_name and org_name != org_id else '(org)'
        choices.append((org_id, label))

    for i, (uid, label) in enumerate(choices, 1):
        num = click.style(f'[{i}]', fg='yellow', bold=True)
        name = click.style(uid, bold=True)
        tag = click.style(label, dim=True)
        click.echo(f'  {num} {name} {tag}')
    click.echo()

    selection = click.prompt('Select user_id', default='1', type=str).strip()

    # Accept number or exact user_id string
    if selection.isdigit() and 1 <= int(selection) <= len(choices):
        return choices[int(selection) - 1][0]
    for uid, _ in choices:
        if selection == uid:
            return uid
    return personal_user_id


def _env_prefix(api_url):
    """'https://api-dev.clarifai.com' -> 'dev', 'https://api-staging...' -> 'staging'."""
    from urllib.parse import urlparse

    host = urlparse(api_url).hostname or ''
    if 'api-dev' in host:
        return 'dev'
    elif 'api-staging' in host:
        return 'staging'
    return 'prod'


@cli.command()
@click.argument('api_url', default=DEFAULT_BASE)
@click.option('--pat', required=False, help='Personal Access Token (skips interactive prompt).')
@click.option(
    '--user-id', required=False, help='User or org ID. Auto-detected from PAT if omitted.'
)
@click.option('--name', required=False, help='Context name. Defaults to the selected user_id.')
@click.pass_context
def login(ctx, api_url, pat, user_id, name):
    """Log in to Clarifai and save credentials.

    \b
    Verifies your PAT, detects your user_id, and saves a named
    context to ~/.config/clarifai/config.

    \b
    API_URL  Clarifai API base URL (default: https://api.clarifai.com).

    \b
    Examples:
      clarifai login                                  # interactive
      clarifai login --pat $MY_PAT                    # non-interactive
      clarifai login --pat $PAT --user-id openai      # org user, skip selection
      clarifai login https://api-dev.clarifai.com     # dev environment
      clarifai login --name my-context                # custom context name
    """
    # 1. Get PAT: --pat flag > CLARIFAI_PAT env var > browser + interactive prompt
    if not pat:
        pat = _get_pat_interactive()

    # 2. Validate PAT + resolve personal user_id
    click.secho('Verifying...', dim=True)
    personal_user_id = _verify_and_resolve_user(pat, api_url)

    # 3. Resolve which user_id to use
    if user_id:
        selected_user_id = user_id
    else:
        orgs = _list_user_orgs(pat, personal_user_id, api_url)
        if orgs:
            selected_user_id = _prompt_user_or_org(personal_user_id, orgs)
        else:
            selected_user_id = personal_user_id

    # 4. Derive context name
    if not name:
        if api_url != DEFAULT_BASE:
            prefix = _env_prefix(api_url)
            name = f"{prefix}-{selected_user_id}"
        else:
            name = selected_user_id

    # 5. Save (update if exists, create if new)
    action = "Updated" if name in ctx.obj.contexts else "Created"
    context = Context(
        name,
        CLARIFAI_API_BASE=api_url,
        CLARIFAI_USER_ID=selected_user_id,
        CLARIFAI_PAT=pat,
    )
    ctx.obj.contexts[name] = context
    ctx.obj.current_context = name
    ctx.obj.to_yaml()

    # 6. Output
    click.echo()
    click.echo(
        click.style('Logged in as ', fg='green')
        + click.style(selected_user_id, fg='green', bold=True)
        + click.style(f' ({api_url})', dim=True)
    )
    click.echo(f'{action} context {click.style(name, bold=True)} and set as active.')


def pat_display(pat):
    return pat[:5] + "****"


def input_or_default(prompt, default):
    value = input(prompt)
    return value if value else default


# Context management commands under config group
@config.command(aliases=['get-contexts', 'list-contexts', 'ls'])
@click.option(
    '-o', '--output-format', default='wide', type=click.Choice(['wide', 'name', 'json', 'yaml'])
)
@click.pass_context
def get_contexts(ctx, output_format):
    """List all available contexts."""
    if output_format == 'wide':
        columns = {
            '': lambda c: '*' if c.name == ctx.obj.current_context else '',
            'NAME': lambda c: c.name,
            'USER_ID': lambda c: c.user_id,
            'API_BASE': lambda c: c.api_base,
            'PAT': lambda c: pat_display(c.pat),
        }
        additional_columns = set()
        for cont in ctx.obj.contexts.values():
            if 'env' in cont:
                for key in cont.to_column_names():
                    if key not in columns:
                        additional_columns.add(key)
        for key in sorted(additional_columns):
            columns[key] = lambda c, k=key: getattr(c, k) if hasattr(c, k) else ""
        formatter = TableFormatter(
            custom_columns=columns,
        )
        print(formatter.format(ctx.obj.contexts.values(), fmt="plain"))
    elif output_format == 'name':
        print('\n'.join(ctx.obj.contexts))
    elif output_format in ('json', 'yaml'):
        dicts = []
        for c, v in ctx.obj.contexts.items():
            context_dict = {}
            d = v.to_serializable_dict()
            d.pop('CLARIFAI_PAT', None)
            context_dict['name'] = c
            context_dict['env'] = d
            dicts.append(context_dict)
        if output_format == 'json':
            print(json.dumps(dicts))
        elif output_format == 'yaml':
            print(yaml.safe_dump(dicts))


@config.command(aliases=['use-context', 'use'])
@click.argument('name', type=str)
@click.pass_context
def use_context(ctx, name):
    """Set the current context."""
    if name not in ctx.obj.contexts:
        raise click.UsageError('Context not found')
    ctx.obj.current_context = name
    ctx.obj.to_yaml()
    print(f'Set {name} as the current context')


@config.command(aliases=['current-context', 'current'])
@click.option('-o', '--output-format', default='name', type=click.Choice(['name', 'json', 'yaml']))
@click.pass_context
def current_context(ctx, output_format):
    """Show the current context's details."""
    if output_format == 'name':
        print(ctx.obj.current_context)
    elif output_format == 'json':
        print(json.dumps(ctx.obj.contexts[ctx.obj.current_context].to_serializable_dict()))
    else:
        print(yaml.safe_dump(ctx.obj.contexts[ctx.obj.current_context].to_serializable_dict()))


@config.command(aliases=['create-context', 'create'])
@click.argument('name')
@click.option(
    '--user-id', required=False, help='User or org ID. Auto-detected from PAT if omitted.'
)
@click.option('--base-url', required=False, default=DEFAULT_BASE, help='API base URL.')
@click.option('--pat', required=False, help='Personal Access Token.')
@click.pass_context
def create_context(ctx, name, user_id, base_url, pat):
    """Create a new named context."""
    if name in ctx.obj.contexts:
        click.secho(
            f'Context "{name}" already exists. Use "clarifai login" to update it.',
            fg='red',
            err=True,
        )
        raise SystemExit(1)

    # Same PAT resolution as login: flag > env > browser + prompt
    if not pat:
        pat = _get_pat_interactive()

    # Same user_id resolution: flag > auto-detect + org selection
    if not user_id:
        click.secho('Verifying...', dim=True)
        personal_user_id = _verify_and_resolve_user(pat, base_url)
        orgs = _list_user_orgs(pat, personal_user_id, base_url)
        if orgs:
            user_id = _prompt_user_or_org(personal_user_id, orgs)
        else:
            user_id = personal_user_id
    else:
        click.secho('Verifying...', dim=True)
        _verify_and_resolve_user(pat, base_url)

    context = Context(name, CLARIFAI_USER_ID=user_id, CLARIFAI_API_BASE=base_url, CLARIFAI_PAT=pat)
    ctx.obj.contexts[name] = context
    ctx.obj.to_yaml()
    click.echo(
        click.style('Context ', fg='green')
        + click.style(name, fg='green', bold=True)
        + click.style(' created ', fg='green')
        + click.style(f'({user_id} @ {base_url})', dim=True)
    )


@config.command(aliases=['e'])
@click.pass_context
def edit(
    ctx,
):
    """Open the configuration file for editing."""
    # For now, just open the config file (not per-context)
    os.system(f'{os.environ.get("EDITOR", "vi")} {ctx.obj.filename}')


@config.command(aliases=['delete-context', 'delete'])
@click.argument('name')
@click.pass_context
def delete_context(ctx, name):
    """Delete a context."""
    if name not in ctx.obj.contexts:
        print(f'{name} is not a valid context')
        sys.exit(1)
    ctx.obj.contexts.pop(name)
    ctx.obj.to_yaml()
    print(f'{name} deleted')


@config.command(aliases=['get-env'])
@click.pass_context
def env(ctx):
    """Print env vars for the active context."""
    ctx.obj.current.print_env_vars()


@config.command(aliases=['show'])
@click.option('-o', '--output-format', default='yaml', type=click.Choice(['json', 'yaml']))
@click.pass_context
def view(ctx, output_format):
    """Display the current configuration."""
    config_dict = {
        'current-context': ctx.obj.current_context,
        'contexts': {
            name: context.to_serializable_dict() for name, context in ctx.obj.contexts.items()
        },
    }

    if output_format == 'json':
        print(json.dumps(config_dict, indent=2))
    else:
        print(yaml.safe_dump(config_dict, default_flow_style=False))


@cli.command()
@click.argument('script', type=str)
@click.option('--context', type=str, help='Context to use')
@click.pass_context
def run(ctx, script, context=None):
    """Execute a script with the current context's environment"""
    context = ctx.obj.current if not context else context
    cmd = f'CLARIFAI_USER_ID={context.user_id} CLARIFAI_API_BASE={context.api_base} CLARIFAI_PAT={context.pat} '
    cmd += ' '.join([f'{k}={v}' for k, v in context.to_serializable_dict().items()])
    cmd += f' {script}'
    os.system(cmd)


# Import the CLI commands to register them
load_command_modules()


def main():
    cli()
