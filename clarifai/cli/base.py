import json
import os
import sys

import click
import yaml

from clarifai.utils.cli import AliasedGroup, TableFormatter, load_command_modules
from clarifai.utils.config import Config, Context
from clarifai.utils.constants import DEFAULT_CONFIG


#@click.group(cls=CustomMultiGroup)
@click.group(cls=AliasedGroup)
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
            'default': {
                'user_id': os.environ.get('CLARIFAI_USER_ID', ''),
                'base_url': os.environ.get('CLARIFAI_API_BASE', 'api.clarifai.com'),
            },
        })
    cfg.to_yaml(config)
    ctx.obj = cfg


@cli.command()
@click.argument('shell', type=click.Choice(['bash', 'zsh']))
def shell_completion(shell):
  """Shell completion script"""
  os.system(f"_CLARIFAI_COMPLETE={shell}_source clarifai")


@cli.group(['cfg'], cls=AliasedGroup)
def config():
  """Manage CLI configuration"""


@config.command(['e'])
@click.pass_context
def edit(ctx):
  """Edit the configuration file"""
  os.system(f'{os.environ.get("EDITOR", "vi")} {ctx.obj.filename}')


@config.command(['current'])
@click.option('-o', '--output-format', default='name', type=click.Choice(['name', 'json', 'yaml']))
@click.pass_context
def current_context(ctx, output_format):
  """Get the current context"""
  if output_format == 'name':
    print(ctx.obj.current_context)
  else:
    if output_format == 'json':
      print(json.dumps(ctx.obj.contexts[ctx.obj.current_context].to_serializable_dict()))
    else:
      print(yaml.safe_dump(ctx.obj.contexts[ctx.obj.current_context].to_serializable_dict()))


@config.command(['list', 'ls'])
@click.option(
    '-o', '--output-format', default='wide', type=click.Choice(['wide', 'name', 'json', 'yaml']))
@click.pass_context
def get_contexts(ctx, output_format):
  """Get all contexts"""
  if output_format == 'wide':
    formatter = TableFormatter(custom_columns={
        '': lambda c: '*' if c.name == ctx.obj.current_context else '',
        'NAME': lambda c: c.name,
        'USER_ID': lambda c: c.user_id,
        'BASE_URL': lambda c: c.base_url,
        'PAT_CONF': lambda c: c.access_token.type,
    })
    print(formatter.format(ctx.obj.contexts.values(), fmt="plain"))
  elif output_format == 'name':
    print('\n'.join(ctx.obj.contexts))
  elif output_format in ('json', 'yaml'):
    dicts = [v.__dict__ for c, v in ctx.obj.contexts.items()]
    for d in dicts:
      d.pop('pat')
    if output_format == 'json':
      print(json.dumps(dicts))
    elif output_format == 'yaml':
      print(yaml.safe_dump(dicts))


@config.command(['use'])
@click.argument('context-name', type=str)
@click.pass_context
def use_context(ctx, context_name):
  """Set the current context"""
  if context_name not in ctx.obj.contexts:
    raise click.UsageError('Context not found')
  ctx.obj.current_context = context_name
  ctx.obj.to_yaml()
  print(f'Set {context_name} as the current context')


@config.command(['cat'])
@click.option('-o', '--output-format', default='yaml', type=click.Choice(['yaml', 'json']))
@click.pass_obj
def dump(ctx_obj, output_format):
  """Dump the configuration to stdout"""
  if output_format == 'yaml':
    yaml.safe_dump(ctx_obj.to_dict(), sys.stdout)
  else:
    json.dump(ctx_obj.to_dict(), sys.stdout, indent=2)


@cli.command()
@click.argument('api_url', default="https://api.clarifai.com")
@click.option('--user_id', required=False, help='User ID')
@click.pass_context
def login(ctx, api_url, user_id):
  """Login command to set PAT and other configurations."""

  name = input('context name (default: "default"): ')
  user_id = user_id if user_id is not None else input('user id: ')
  access_token_type = input('access token type (env or raw, default: "env"): ')
  access_token_value = input('access token value (default: "CLARIFAI_PAT"): ')

  if access_token_type != '':
    access_token = dict(type=access_token_type, value=access_token_value)
    context = Context(
        name=name,
        base_url=api_url,
        user_id=user_id,
        access_token=access_token,
    )
  else:
    context = Context(
        name=name,
        base_url=api_url,
        user_id=user_id,
    )
    if access_token_value != '':
      context.access_token['value'] = access_token_value

  if context.name == '':
    context.name = 'default'

  ctx.obj.contexts[context.name] = context
  ctx.obj.current_context = context.name

  ctx.obj.to_yaml()


@cli.group(cls=AliasedGroup)
def context():
  """Manage contexts"""


@context.command(['ls'])
@click.pass_context
def list(ctx):
  """List available contexts"""
  formatter = TableFormatter(custom_columns={
      '': lambda c: '*' if c.name == ctx.obj.current_context else '',
      'NAME': lambda c: c.name,
      'USER_ID': lambda c: c.user_id,
      'BASE_URL': lambda c: c.base_url,
      'PAT_CONF': lambda c: str(c.access_token)
  })
  print(formatter.format(ctx.obj.contexts.values(), fmt="plain"))


def input_or_default(prompt, default):
  value = input(prompt)
  return value if value else default


@context.command()
@click.argument('name')
@click.option('--user-id', required=False, help='User ID')
@click.option('--base-url', required=False, help='Base URL')
@click.option('--access-token-type', required=False, help='Access token type')
@click.option('--access-token-value', required=False, help='Access token value')
@click.pass_context
def create(
    ctx,
    name,
    user_id=None,
    base_url=None,
    access_token_type=None,
    access_token_value=None,
):
  """Create a new context"""
  if name in ctx.obj.contexts:
    print(f'{name} already exists')
    exit(1)
  if not user_id:
    user_id = input('user id: ')
  if not base_url:
    base_url = input_or_default('base url (default: https://api.clarifai.com): ',
                                'https://api.clarifai.com')
  if not access_token_type:
    access_token_type = input_or_default('access token type (env or raw, default: "env"): ', 'env')
  if not access_token_value:
    access_token_value = input_or_default('access token value (default: "CLARIFAI_PAT"): ',
                                          'CLARIFAI_PAT')

  context = Context(
      name=name,
      user_id=user_id,
      base_url=base_url,
      access_token=dict(type=access_token_type, value=access_token_value))
  ctx.obj.contexts[context.name] = context
  ctx.obj.to_yaml()


# write a click command to delete a context
@context.command(['rm'])
@click.argument('name')
@click.pass_context
def delete(ctx, name):
  """Delete a context"""
  if name not in ctx.obj.contexts:
    print(f'{name} is not a valid context')
    exit(1)
  ctx.obj.contexts.pop(name)
  ctx.obj.to_yaml()
  print(f'{name} deleted')


@context.command()
@click.argument('name', type=str)
@click.pass_context
def use(ctx, name):
  """Set the current context"""
  if name not in ctx.obj.contexts:
    raise click.UsageError('Context not found')
  ctx.obj.current_context = name
  ctx.obj.to_yaml()
  print(f'Set {name} as the current context')


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
