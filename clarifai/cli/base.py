import json
import os
import sys

import click
import yaml

from clarifai import __version__
from clarifai.utils.cli import (
    LazyAliasedGroup,
    TableFormatter,
    masked_input,
)
from clarifai.utils.config import Config, Context
from clarifai.utils.constants import DEFAULT_BASE, DEFAULT_CONFIG, DEFAULT_UI
from clarifai.utils.logging import logger


@click.group(cls=LazyAliasedGroup)
@click.version_option(version=__version__)
@click.option('--config', default=DEFAULT_CONFIG, help='Path to config file')
@click.option('--context', default=None, help='Context to use for this command')
@click.pass_context
def cli(ctx, config, context):
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

    # Store the context override in Click context for all commands to access
    if context:
        validate_and_get_context(ctx.obj, context)
        ctx.obj.context_override = context


@cli.command()
@click.argument('shell', type=click.Choice(['bash', 'zsh']))
def shell_completion(shell):
    """Shell completion script"""
    os.system(f"_CLARIFAI_COMPLETE={shell}_source clarifai")


@cli.group(cls=LazyAliasedGroup)
def config():
    """
    Manage multiple configuration profiles (contexts).

    Authentication Precedence:\n
      1. Environment variables (e.g., `CLARIFAI_PAT`) are used first if set.
      2. The settings from the active context are used if no environment variables are provided.\n
    """


@cli.command()
@click.argument('api_url', default=DEFAULT_BASE)
@click.option('--user_id', required=False, help='User ID')
@click.pass_context
def login(ctx, api_url, user_id):
    """Login command to set PAT and other configurations."""
    from clarifai.utils.cli import validate_context_auth

    # Input user_id if not supplied
    if not user_id:
        user_id = click.prompt('Enter your Clarifai user ID', type=str)

    click.echo()  # Blank line for readability

    # Check for environment variable first
    env_pat = os.environ.get('CLARIFAI_PAT')
    if env_pat:
        use_env = click.confirm('Use CLARIFAI_PAT from environment?', default=True)
        if use_env:
            pat = env_pat
        else:
            click.echo(f'> Create a PAT at: https://clarifai.com/{user_id}/settings/secrets')
            pat = masked_input('Enter your Personal Access Token (PAT): ')
    else:
        click.echo('> To authenticate, you\'ll need a Personal Access Token (PAT).')
        click.echo(f'> Create one at: https://clarifai.com/{user_id}/settings/secrets')
        click.echo('> Tip: Set CLARIFAI_PAT environment variable to skip this prompt.\n')
        pat = masked_input('Enter your Personal Access Token (PAT): ')

    # Progress indicator
    click.echo('\n> Verifying token...')
    validate_context_auth(pat, user_id, api_url)

    # Save context with default name
    context_name = 'default'
    context = Context(
        context_name,
        CLARIFAI_API_BASE=api_url,
        CLARIFAI_USER_ID=user_id,
        CLARIFAI_PAT=pat,
    )

    ctx.obj.contexts[context_name] = context
    ctx.obj.current_context = context_name
    ctx.obj.to_yaml()
    click.secho(f'✅ Success! You\'re logged in as {user_id}', fg='green')
    click.echo('💡 Tip: Use `clarifai config` to manage multiple accounts or environments')

    logger.info(f"Login successful for user '{user_id}' in context '{context_name}'")


@cli.command()
@click.pass_context
def whoami(ctx):
    """Display information about the current user."""
    from clarifai_grpc.grpc.api.status import status_code_pb2

    from clarifai.client.user import User

    # Get the current context
    cfg = ctx.obj
    current_ctx = cfg.contexts[cfg.current_context]

    # Get user_id from context
    context_user_id = current_ctx.CLARIFAI_USER_ID
    pat = current_ctx.CLARIFAI_PAT
    base_url = current_ctx.CLARIFAI_API_BASE

    # Display context user info
    click.echo("Context User ID: " + click.style(context_user_id, fg='cyan', bold=True))

    # Call GetUser RPC with "me" to get the actual authenticated user
    try:
        user_client = User(user_id="me", pat=pat, base_url=base_url)
        response = user_client.get_user_info(user_id="me")

        if response.status.code == status_code_pb2.SUCCESS:
            actual_user_id = response.user.id
            click.echo(
                "Authenticated User ID: " + click.style(actual_user_id, fg='green', bold=True)
            )

            # Check if they differ
            if context_user_id != actual_user_id:
                click.echo()
                click.secho(
                    "��️  Warning: The context user ID differs from the authenticated user ID!",
                    fg='yellow',
                )
                click.echo(
                    "This means you as the caller will be calling different user or organization."
                )
        else:
            click.secho(f"Error getting user info: {response.status.description}", fg='red')

    except Exception as e:
        click.secho(f"Error: Could not retrieve authenticated user info: {str(e)}", fg='red')


def _warn_env_pat():
    """Warn if CLARIFAI_PAT environment variable is still set."""
    if os.environ.get('CLARIFAI_PAT'):
        click.echo()
        click.secho("Warning: CLARIFAI_PAT environment variable is still set.", fg='yellow')
        click.echo(
            "   Run `unset CLARIFAI_PAT` (Linux/Mac) or `$env:CLARIFAI_PAT = ''` (PowerShell) to fully log out."
        )


def _clear_context_pat(context):
    """Clear the PAT from a context. Returns True if a PAT was actually cleared."""
    # Access the config dict directly so env-var precedence does not interfere.
    pat = context['env'].get('CLARIFAI_PAT', '')
    if pat:
        context['env']['CLARIFAI_PAT'] = ''
        return True
    return False


def _logout_one_context(cfg, name, delete=False):
    """Clear (and optionally delete) a single context.

    Handles last-context protection, auto-switching, persistence, and messaging.
    """
    ctx_obj = cfg.contexts.get(name)
    if not ctx_obj:
        if not cfg.contexts:
            raise click.ClickException("No contexts are configured. Run `clarifai login` first.")
        available = ', '.join(cfg.contexts.keys())
        raise click.ClickException(f"Context '{name}' not found. Available: {available}")

    if delete:
        if len(cfg.contexts) <= 1:
            if _clear_context_pat(ctx_obj):
                cfg.to_yaml()
                click.secho(
                    f"Cleared credentials for '{name}' (kept as it is the only context).",
                    fg='green',
                )
            else:
                click.echo(
                    f"Already logged out of context '{name}' (kept as it is the only context)."
                )
        else:
            cfg.contexts.pop(name)
            if cfg.current_context == name:
                cfg.current_context = next(iter(cfg.contexts))
                click.echo(f"Switched to context '{cfg.current_context}'.")
            cfg.to_yaml()
            click.secho(f"Context '{name}' deleted.", fg='green')
    elif _clear_context_pat(ctx_obj):
        cfg.to_yaml()
        click.secho(f"Logged out of context '{name}'.", fg='green')
    else:
        click.echo(f"Already logged out of context '{name}'.")


def _logout_all_contexts(cfg):
    """Clear PATs from every context, persist, and print a summary.

    Returns the number of contexts that were actually cleared.
    """
    cleared = sum(1 for ctx in cfg.contexts.values() if _clear_context_pat(ctx))
    cfg.to_yaml()
    if cleared:
        click.secho(f"Logged out of all contexts ({cleared} cleared).", fg='green')
    else:
        click.echo("Already logged out of all contexts.")
    return cleared


@cli.command()
@click.option(
    '--current',
    'flag_current',
    is_flag=True,
    default=False,
    help='Clear credentials from the current context (non-interactive).',
)
@click.option(
    '--all',
    'flag_all',
    is_flag=True,
    default=False,
    help='Clear credentials from all contexts (non-interactive).',
)
@click.option(
    '--context',
    'flag_context',
    default=None,
    type=str,
    help='Clear credentials from a specific named context (non-interactive).',
)
@click.option(
    '--delete',
    'flag_delete',
    is_flag=True,
    default=False,
    help='Also delete the context entry (use with --current or --context).',
)
@click.pass_context
def logout(ctx, flag_current, flag_all, flag_context, flag_delete):
    """Log out by clearing saved credentials.

    Without flags, an interactive menu is shown. Use flags for
    programmatic / non-interactive usage.

    \b
    Examples:
      clarifai logout                        # Interactive
      clarifai logout --current              # Clear current context PAT
      clarifai logout --context staging      # Clear 'staging' PAT
      clarifai logout --context staging --delete  # Remove 'staging' entirely
      clarifai logout --all                  # Clear every context PAT
    """
    cfg = ctx.obj
    if not cfg or not hasattr(cfg, 'contexts'):
        raise click.ClickException("Not logged in. Run `clarifai login` first.")

    # --- Validation for flag combinations ---
    if flag_all and (flag_current or flag_context or flag_delete):
        raise click.UsageError("--all cannot be combined with --current, --context, or --delete.")

    if flag_delete and not (flag_current or flag_context):
        raise click.UsageError("--delete requires --current or --context.")

    if flag_current and flag_context:
        raise click.UsageError("Cannot use --current and --context together.")

    # --- Non-interactive paths ---
    if flag_all:
        _logout_all_contexts(cfg)
        _warn_env_pat()
        return

    if flag_current:
        _logout_one_context(cfg, cfg.current_context, delete=flag_delete)
        _warn_env_pat()
        return

    if flag_context:
        _logout_one_context(cfg, flag_context, delete=flag_delete)
        _warn_env_pat()
        return

    # --- Interactive flow ---
    cur_name = cfg.current_context
    cur_ctx = cfg.contexts.get(cur_name)
    if not cur_ctx:
        raise click.ClickException("No active context found. Run `clarifai login` first.")

    user_id = cur_ctx['env'].get('CLARIFAI_USER_ID', 'unknown')
    api_base = cur_ctx['env'].get('CLARIFAI_API_BASE', DEFAULT_BASE)
    click.echo(
        f"\nCurrent context is configured for user '{user_id}' (context: '{cur_name}', api: {api_base})\n"
    )

    # Build menu
    choices = []
    choices.append(('switch', 'Switch to another context'))
    choices.append(('logout_current', 'Log out of current context (clear credentials)'))
    choices.append(('logout_delete', 'Log out and delete current context'))
    choices.append(('logout_all', 'Log out of all contexts'))
    choices.append(('cancel', 'Cancel'))

    for i, (_, label) in enumerate(choices, 1):
        click.echo(f"  {i}. {label}")

    click.echo()
    choice_num = click.prompt('Enter choice', type=click.IntRange(1, len(choices)))
    action = choices[choice_num - 1][0]

    if action == 'cancel':
        click.echo('Cancelled. No changes made.')
        return

    if action == 'switch':
        other_contexts = [n for n in cfg.contexts if n != cur_name]
        if not other_contexts:
            click.echo("No other contexts available. Use `clarifai login` to create one.")
            return
        click.echo('\nAvailable contexts:')
        for i, name in enumerate(other_contexts, 1):
            uid = cfg.contexts[name]['env'].get('CLARIFAI_USER_ID', 'unknown')
            click.echo(f"  {i}. {name} (user: {uid})")
        click.echo()
        idx = click.prompt('Switch to', type=click.IntRange(1, len(other_contexts)))
        target_name = other_contexts[idx - 1]
        cfg.current_context = target_name
        cfg.to_yaml()
        click.secho(
            f"Switched to context '{target_name}'. No credentials were cleared.", fg='green'
        )
        return

    if action == 'logout_current':
        _logout_one_context(cfg, cur_name)

    elif action == 'logout_delete':
        _logout_one_context(cfg, cur_name, delete=True)

    elif action == 'logout_all':
        _logout_all_contexts(cfg)

    _warn_env_pat()
    click.echo("\nRun 'clarifai login' to re-authenticate.")


def pat_display(pat):
    return pat[:5] + "****"


def input_or_default(prompt, default):
    value = input(prompt)
    return value if value else default


def validate_and_get_context(config, context_name):
    """Validate that a context exists and return it.

    Args:
        config: Config object containing contexts
        context_name: Name of the context to validate

    Returns:
        Context object if found

    Raises:
        click.UsageError: If context is not found
    """
    if context_name not in config.contexts:
        raise click.UsageError(
            f"Context '{context_name}' not found. Available contexts: {', '.join(config.contexts.keys())}"
        )
    return config.contexts[context_name]


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
@click.option('--user-id', required=False, help='User ID')
@click.option('--base-url', required=False, help='Base URL')
@click.option('--pat', required=False, help='Personal access token')
@click.pass_context
def create_context(
    ctx,
    name,
    user_id=None,
    base_url=None,
    pat=None,
):
    """Create a new context."""
    from clarifai.utils.cli import validate_context_auth

    if name in ctx.obj.contexts:
        click.secho(f'Error: Context "{name}" already exists', fg='red', err=True)
        sys.exit(1)
    if not user_id:
        user_id = input('user id: ')
    if not base_url:
        base_url = input_or_default(
            'base url (default: https://api.clarifai.com): ', 'https://api.clarifai.com'
        )
    if not pat:
        # Check for environment variable first
        env_pat = os.environ.get('CLARIFAI_PAT')
        if env_pat:
            use_env = click.confirm('Found CLARIFAI_PAT in environment. Use it?', default=True)
            if use_env:
                pat = env_pat
            else:
                pat = masked_input('Enter your Personal Access Token (PAT): ')
        else:
            click.echo('Tip: Set CLARIFAI_PAT environment variable to skip this step.')
            pat = masked_input('Enter your Personal Access Token (PAT): ')
    validate_context_auth(pat, user_id, base_url)
    context = Context(name, CLARIFAI_USER_ID=user_id, CLARIFAI_API_BASE=base_url, CLARIFAI_PAT=pat)
    ctx.obj.contexts[context.name] = context
    ctx.obj.to_yaml()
    click.secho(f"✅ Context '{name}' created successfully", fg='green')


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
    # Get the effective context - either from --context flag or current context
    if context:
        context_obj = validate_and_get_context(ctx.obj, context)
    else:
        context_obj = ctx.obj.current

    cmd = f'CLARIFAI_USER_ID={context_obj.user_id} CLARIFAI_API_BASE={context_obj.api_base} CLARIFAI_PAT={context_obj.pat} '
    cmd += ' '.join([f'{k}={v}' for k, v in context_obj.to_serializable_dict().items()])
    cmd += f' {script}'
    os.system(cmd)


# Import the CLI commands to register them
# load_command_modules() - Now handled lazily by LazyLazyAliasedGroupp


def main():
    cli()
