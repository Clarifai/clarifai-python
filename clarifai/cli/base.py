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
@click.option('--config', default=DEFAULT_CONFIG, help='Path to config file.')
@click.option('--context', default=None, help='Context to use for this command')
@click.pass_context
def cli(ctx, config, context):
    """Build, deploy, and manage AI models on the Clarifai platform."""
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


@cli.command(short_help='Generate shell completion script.')
@click.argument('shell', type=click.Choice(['bash', 'zsh']))
def shell_completion(shell):
    """Generate shell completion script for bash or zsh."""
    os.system(f"_CLARIFAI_COMPLETE={shell}_source clarifai")


@cli.group(cls=LazyAliasedGroup, short_help='Manage configuration profiles (contexts).')
def config():
    """Manage configuration profiles (contexts).

    \b
    Authentication Precedence:
      1. Environment variables (e.g., CLARIFAI_PAT) are used first if set.
      2. The active context settings are used as fallback.
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


@cli.command(short_help='Authenticate and save credentials.')
@click.argument('api_url', default=DEFAULT_BASE)
@click.option('--pat', required=False, help='Personal Access Token (skips interactive prompt).')
@click.option(
    '--user-id', required=False, help='User or org ID. Auto-detected from PAT if omitted.'
)
@click.option('--name', required=False, help='Context name. Defaults to the selected user_id.')
@click.pass_context
def login(ctx, api_url, pat, user_id, name):
    """Authenticate and save credentials.

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


@cli.command(short_help='Show current user and context.')
@click.option('--orgs', is_flag=True, help='Show organizations you belong to.')
@click.option('--all', 'show_all', is_flag=True, help='Show full profile (email, name, orgs).')
@click.option(
    '-o',
    '--output-format',
    default='wide',
    type=click.Choice(['wide', 'json']),
    help='Output format.',
)
@click.pass_context
def whoami(ctx, orgs, show_all, output_format):
    """Show current user and context.

    \b
    Examples:
      clarifai whoami                # user + context (local only)
      clarifai whoami --orgs         # include organizations (API call)
      clarifai whoami --all          # full profile + orgs (API call)
      clarifai whoami -o json        # JSON output for scripting
    """
    cfg = ctx.obj
    context = cfg.current

    # Check if logged in
    pat = context.get('pat')
    user_id = context.get('user_id')
    api_base = context.get('api_base', DEFAULT_BASE)

    if not pat or not user_id or user_id == '_empty_':
        click.secho("Not logged in. Run 'clarifai login' to authenticate.", fg='red', err=True)
        raise SystemExit(1)

    data = {
        'user_id': user_id,
        'context': cfg.current_context,
        'api_base': api_base,
    }

    # Fetch full profile and/or orgs if requested
    org_list = []
    if show_all or orgs:
        try:
            org_list = _list_user_orgs(pat, user_id, api_base)
            data['organizations'] = [{'id': oid, 'name': oname} for oid, oname in org_list]
        except Exception:
            org_list = []

    if show_all:
        try:
            from clarifai.client.user import User

            user = User(user_id='me', pat=pat, base_url=api_base)
            response = user.get_user_info(user_id=user_id)
            u = response.user
            if u.full_name:
                data['name'] = u.full_name
            if u.primary_email:
                data['email'] = u.primary_email
            if u.company_name:
                data['company'] = u.company_name
        except Exception:
            if output_format == 'wide':
                click.secho(
                    'Warning: could not fetch full profile from API.', fg='yellow', err=True
                )

    # Output
    if output_format == 'json':
        click.echo(json.dumps(data))
    else:
        click.echo(
            click.style('User:     ', bold=True) + click.style(user_id, fg='green', bold=True)
        )
        if data.get('name'):
            click.echo(click.style('Name:     ', bold=True) + data['name'])
        if data.get('email'):
            click.echo(click.style('Email:    ', bold=True) + data['email'])
        if data.get('company'):
            click.echo(click.style('Company:  ', bold=True) + data['company'])
        click.echo(
            click.style('Context:  ', bold=True)
            + f'{cfg.current_context} @ '
            + click.style(api_base, dim=True)
        )

        if org_list:
            click.echo()
            click.echo(click.style('Organizations:', bold=True))
            for org_id, org_name in org_list:
                click.echo(
                    f'  {click.style(org_id, bold=True)}   {click.style(org_name, dim=True)}'
                )


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
            '': lambda c: click.style('*', fg='green', bold=True)
            if c.name == ctx.obj.current_context
            else '',
            'NAME': lambda c: click.style(c.name, bold=True)
            if c.name == ctx.obj.current_context
            else c.name,
            'USER_ID': lambda c: c.user_id,
            'API_BASE': lambda c: click.style(c.api_base, dim=True),
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
        raise click.UsageError(f'Context "{name}" not found')
    ctx.obj.current_context = name
    ctx.obj.to_yaml()
    click.echo(
        click.style('Switched to context ', fg='green') + click.style(name, fg='green', bold=True)
    )


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
        click.secho(f'Context "{name}" not found.', fg='red', err=True)
        sys.exit(1)
    ctx.obj.contexts.pop(name)
    ctx.obj.to_yaml()
    click.echo(
        click.style('Deleted context ', fg='yellow') + click.style(name, fg='yellow', bold=True)
    )


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


@cli.command(short_help='Run a script with context env vars.')
@click.argument('script', type=str)
@click.option('--context', type=str, help='Context to use')
@click.pass_context
def run(ctx, script, context=None):
    """Run a script with the current context's environment variables injected."""
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

# Define section ordering for `clarifai --help`
cli.command_sections = [
    ('Auth', ['login', 'whoami']),
    ('Config', ['config']),
    ('Models', ['model']),
    ('Pipelines', ['pipeline', 'pipeline-step', 'pipelinerun', 'pipelinetemplate']),
    ('Compute', ['computecluster', 'nodepool', 'deployment']),
    ('Other', ['artifact', 'run', 'shell-completion']),
]


def main():
    cli()
