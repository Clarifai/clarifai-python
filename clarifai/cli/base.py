import json
import os
import sys

import click
import yaml

from clarifai import __version__
from clarifai.utils.cli import AliasedGroup, TableFormatter, load_command_modules
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

    click.echo('> To authenticate, you\'ll need a Personal Access Token (PAT).')
    click.echo(
        f'> You can create one from your account settings: https://clarifai.com/{user_id}/settings/security\n'
    )

    # Securely input PAT
    pat = input_or_default(
        'Enter your Personal Access Token (PAT) value (or type "ENVVAR" to use an environment variable): ',
        'ENVVAR',
    )
    if pat.lower() == 'envvar':
        pat = os.environ.get('CLARIFAI_PAT')
        if not pat:
            logger.error(
                'Environment variable "CLARIFAI_PAT" not set. Please set it in your terminal.'
            )
            click.echo(
                'Aborting login. Please set the environment variable or provide a PAT value and try again.'
            )
            click.abort()
    # Progress indicator
    click.echo('\n> Verifying token...')
    validate_context_auth(pat, user_id, api_url)

    # Context naming
    default_context_name = 'default'
    click.echo('\n> Let\'s save these credentials to a new context.')
    click.echo('> You can have multiple contexts to easily switch between accounts or projects.\n')
    context_name = click.prompt("Enter a name for this context", default=default_context_name)

    # Save context
    context = Context(
        context_name,
        CLARIFAI_API_BASE=api_url,
        CLARIFAI_USER_ID=user_id,
        CLARIFAI_PAT=pat,
    )

    ctx.obj.contexts[context_name] = context
    ctx.obj.current_context = context_name
    ctx.obj.to_yaml()
    click.secho('✅ Success! You are now logged in.', fg='green')
    click.echo(f'Credentials saved to the \'{context_name}\' context.\n')
    click.echo('💡 To switch contexts later, use `clarifai config use-context <name>`.')

    logger.info(f"Login successful for user '{user_id}' in context '{context_name}'")


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
        logger.info(f'"{name}" context already exists')
        sys.exit(1)
    if not user_id:
        user_id = input('user id: ')
    if not base_url:
        base_url = input_or_default(
            'base url (default: https://api.clarifai.com): ', 'https://api.clarifai.com'
        )
    if not pat:
        pat = input_or_default(
            'personal access token value (default: "ENVVAR" to get our of env var rather than config): ',
            'ENVVAR',
        )
    if pat.lower() == 'envvar':
        pat = os.environ.get('CLARIFAI_PAT')
        if not pat:
            logger.error(
                'Environment variable "CLARIFAI_PAT" not set. Please set it in your terminal.'
            )
            click.echo(
                'Aborting context creation. Please set the environment variable or provide a PAT value and try again.'
            )
            click.abort()
    validate_context_auth(pat, user_id, base_url)
    context = Context(name, CLARIFAI_USER_ID=user_id, CLARIFAI_API_BASE=base_url, CLARIFAI_PAT=pat)
    ctx.obj.contexts[context.name] = context
    ctx.obj.to_yaml()
    logger.info(f"Context '{name}' created successfully")


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
    cmd = f'CLARIFAI_USER_ID={context.user_id} CLARIFAI_API_BASE={context.base_url} CLARIFAI_PAT={context.pat} '
    cmd += ' '.join([f'{k}={v}' for k, v in context.env.items()])
    cmd += f' {script}'
    os.system(cmd)


# Import the CLI commands to register them
load_command_modules()


def main():
    cli()
