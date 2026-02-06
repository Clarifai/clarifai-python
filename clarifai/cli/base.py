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


@click.group(cls=AliasedGroup, invoke_without_command=True)
@click.version_option(version=__version__)
@click.option('--config', default=DEFAULT_CONFIG)
@click.pass_context
def cli(ctx, config):
    """Clarifai CLI - Chat is the default command.

    Run `clarifai` to start the chat interface, or use subcommands like `clarifai config`, `clarifai login`, etc.
    """
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
            click.echo(f'> Create a PAT at: https://clarifai.com/{user_id}/settings/security')
            pat = masked_input('Enter your Personal Access Token (PAT): ')
    else:
        click.echo('> To authenticate, you\'ll need a Personal Access Token (PAT).')
        click.echo(f'> Create one at: https://clarifai.com/{user_id}/settings/security')
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
def logout(ctx):
    """Logout from the current context by clearing the PAT."""
    current_context_name = ctx.obj.current_context
    current_context = ctx.obj.current

    if current_context.name == '_empty_' or not current_context.pat:
        click.secho("You are not currently logged in.", fg='yellow')
        return

    # Clear PAT from current context
    current_context['env']['CLARIFAI_PAT'] = ''
    ctx.obj.to_yaml()

    click.secho(f"✅ Successfully logged out from '{current_context_name}'.", fg='green')
    click.echo("Your PAT has been cleared from this context.")
    logger.info(f"Logout successful from context '{current_context_name}'")


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


@config.command(aliases=['s'])
@click.argument('key', type=str)
@click.argument('value', type=str)
@click.option(
    '--context',
    '-c',
    type=str,
    default=None,
    help='Context to set the value in. Defaults to current context.',
)
@click.pass_context
def set(ctx, key, value, context):
    """Set a configuration value in the current or specified context.

    Supported keys:
        chat_model_url  - URL of the model to use for `clarifai chat`

    Examples:
        clarifai config set chat_model_url https://clarifai.com/openai/chat-completion/models/gpt-4o
        clarifai config set chat_model_url https://clarifai.com/my-org/my-app/models/my-model -c my-context
    """
    # Determine which context to modify
    context_name = context if context else ctx.obj.current_context

    if context_name not in ctx.obj.contexts:
        click.secho(f"Error: Context '{context_name}' not found.", fg='red')
        sys.exit(1)

    target_context = ctx.obj.contexts[context_name]

    # Set the value in the context's env dict
    # Use CLARIFAI_ prefix for consistency with other config values
    env_key = f'CLARIFAI_{key.upper()}' if not key.upper().startswith('CLARIFAI_') else key.upper()
    target_context['env'][env_key] = value

    ctx.obj.to_yaml()
    click.secho(f"✓ Set '{key}' = '{value}' in context '{context_name}'", fg='green')


@config.command(aliases=['g'])
@click.argument('key', type=str)
@click.option(
    '--context',
    '-c',
    type=str,
    default=None,
    help='Context to get the value from. Defaults to current context.',
)
@click.pass_context
def get(ctx, key, context):
    """Get a configuration value from the current or specified context.

    Examples:
        clarifai config get chat_model_url
        clarifai config get pat -c my-context
    """
    # Determine which context to read from
    context_name = context if context else ctx.obj.current_context

    if context_name not in ctx.obj.contexts:
        click.secho(f"Error: Context '{context_name}' not found.", fg='red')
        sys.exit(1)

    target_context = ctx.obj.contexts[context_name]

    # Try to get the value
    value = target_context.get(key)
    if value is not None:
        # Mask PAT values for security
        if key.lower() in ('pat', 'token', 'secret'):
            display_value = value[:5] + '****' if len(value) > 5 else '****'
        else:
            display_value = value
        click.echo(display_value)
    else:
        click.secho(f"Key '{key}' not found in context '{context_name}'", fg='yellow')
        sys.exit(1)


@config.command(aliases=['get-env'])
@click.pass_context
def env(ctx):
    """Print env vars for the active context."""
    ctx.obj.current.print_env_vars()


@config.command(aliases=['show'])
@click.option('-o', '--output-format', default='yaml', type=click.Choice(['json', 'yaml']))
@click.pass_context
def view(ctx, output_format):
    """Display the current configuration with defaults."""
    from clarifai.cli.chat import DEFAULT_CHAT_MODEL_URL

    contexts_dict = {}
    for name, context in ctx.obj.contexts.items():
        context_data = context.to_serializable_dict()
        # Add defaults for known configurable values if not explicitly set
        if 'CLARIFAI_CHAT_MODEL_URL' not in context_data:
            context_data['CLARIFAI_CHAT_MODEL_URL'] = f'{DEFAULT_CHAT_MODEL_URL} (default)'
        contexts_dict[name] = context_data

    config_dict = {
        'current-context': ctx.obj.current_context,
        'contexts': contexts_dict,
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


@cli.result_callback()
@click.pass_context
def default_command(ctx, *args, **kwargs):
    """If no subcommand is provided, run chat as the default."""
    if ctx.invoked_subcommand is None:
        # Import here to avoid circular imports
        from clarifai.cli.chat import chat

        # Invoke the chat command
        ctx.invoke(chat)


def main():
    cli()
