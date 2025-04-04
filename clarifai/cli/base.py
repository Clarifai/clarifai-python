import click
import sys
import os
import typing as t
import yaml

from collections import defaultdict
from dataclasses import dataclass, field
from ..utils.cli import
from clarifai.utils.cli import dump_yaml, from_yaml, load_command_modules, set_base_url, dump_yaml, from_yaml, load_command_modules, TableFormatter, AliasedGroup
from clarifai.utils.logging import logger

@dataclass
class AccessToken():
  type: str
  value: str

  def __str__(self):
    return f'{self.type}:{self.value}' if self.type == 'env' else '********'

  def to_serializable_dict(self):
    return self.__dict__

  @classmethod
  def from_serializable_dict(cls, _dict):
    return cls(**_dict)

@dataclass
class Context():
  name: str
  user_id: str
  base_url: str
  access_token: AccessToken = field(default_factory=lambda: AccessToken('env', 'CLARIFAI_PAT'))
  env: t.Dict[str, str] = field(default_factory=dict)

  pat: str = None

  def _resolve_pat(self) -> str:
    if self.access_token['type'].lower() == 'env':
      return os.getenv(self.access_token['value'], '')
    elif self.access_token['type'].lower() == 'raw':
      return self.access_token['value']
    else:
      raise Exception('Only "env" and "raw" methods are supported')

  def __post_init__(self):
    self.pat = self._resolve_pat()
    self.access_token = AccessToken(**self.access_token)

  def to_serializable_dict(self):
    result = {
      'name': self.name,
      'user_id': self.user_id,
      'base_url': self.base_url,
      'access_token': self.access_token.to_serializable_dict(),
    }
    if self.env:
      result['env'] = self.env
    return result


@dataclass
class Config():
  current_context: str
  filename: str
  contexts: dict[str, Context] = field(default_factory=dict)

  def __post_init__(self):
    for k, v in self.contexts.items():
      if 'name' not in v:
        v['name'] = k
    self.contexts = {k: Context(**v) for k, v in self.contexts.items()}

  @classmethod
  def from_yaml(cls, filename: str = None):
    with open(filename, 'r') as f:
      cfg = yaml.safe_load(f)
    return cls(**cfg, filename=filename)

  def to_dict(self):
    return {
              'current_context': self.current_context,
              'contexts': {
                  k: v.to_serializable_dict()
                  for k, v in self.contexts.items()
              }
          }

  def to_yaml(self, filename: str = None):
    if filename is None:
      filename = self.filename
    dir = os.path.dirname(filename)
    if len(dir):
      os.makedirs(dir, exist_ok=True)
    _dict = self.to_dict()
    for k, v in _dict['contexts'].items():
      v.pop('name', None)
    with open(filename, 'w') as f:
      yaml.safe_dump(_dict, f)

#@click.group(cls=CustomMultiGroup)
@click.group(cls=AliasedGroup)
@click.option('--config', default=f'{os.environ["HOME"]}/.config/clarifai/config')
@click.pass_context
def cli(ctx, config):
  """Clarifai CLI"""
  ctx.ensure_object(dict)
  if os.path.exists(config):
    cfg = Config.from_yaml(filename=config)
    ctx.obj = cfg
  else:
    cfg = Config(filename=config,
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
@click.argument('shell', type=click.Choice(['bash','zsh']))
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

@config.command()
@click.option('-o', '--output-format', default='name', type=click.Choice(['name','json', 'yaml']))
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


@config.command()
@click.option('-o',
              '--output-format',
              default='wide',
              type=click.Choice(['wide', 'name', 'json', 'yaml']))
@click.pass_context
def get_contexts(ctx, output_format):
  """Get all contexts"""
  if output_format == 'wide':
    formatter = TableFormatter(
        custom_columns={
            '': lambda c: '*' if c.name == ctx.obj.current_context else '',
            'NAME': lambda c: c.name,
            'USER_ID': lambda c: c.user_id,
            'BASE_URL': lambda c: c.base_url,
            'PAT_CONF': lambda c: c.access_token.get('type', 'default')
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
@click.option('-o', '--output-format', default='yaml', type=click.Choice(['yaml','json']))
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

  name=input('context name (default: "default"): ')
  user_id=user_id if user_id is not None else input('user id: ')
  access_token_type=input('access token type (env or raw, default: "env"): ')
  access_token_value=input('access token value (default: "CLARIFAI_PAT"): ')

  if access_token_type != '':
    access_token = dict(
      type=access_token_type,
      value=access_token_value
    )
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
  formatter = TableFormatter(
      custom_columns={
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
    base_url = input_or_default('base url (default: https://api.clarifai.com): ', 'https://api.clarifai.com')
  if not access_token_type:
    access_token_type = input_or_default('access token type (env or raw, default: "env"): ', 'env')
  if not access_token_value:
    access_token_value = input_or_default('access token value (default: "CLARIFAI_PAT"): ', 'CLARIFAI_PAT')

  context = Context(
    name=name,
    user_id=user_id,
    base_url=base_url,
    access_token=dict(
      type=access_token_type,
      value=access_token_value
    )
  )
  ctx.obj.contexts[context.name] = context
  ctx.obj.to_yaml()

@context.command(['e'])
@click.argument('name')
@click.pass_context
def edit(ctx, name):
  """Edit a config"""
  context = ctx.obj.contexts.get(name, None)
  if context is None:
    print(f'{name} is not a valid context')
    exit(1)

  import tempfile

  with tempfile.NamedTemporaryFile(mode='w', delete=True) as f:
    yaml.dump(context.to_serializable_dict(), f)
    os.system(f'{os.environ.get("EDITOR", "vi")} {f.name}')
    with open(f.name, 'r') as f:
      ctx.obj.contexts.pop(name)
      new_context = Context(**yaml.safe_load(f))
      ctx.obj.contexts[new_context.name] = new_context
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
  context = ctx.obj.contexts[ctx.obj.current_context if not context else context]
  cmd = f'CLARIFAI_USER_ID={context.user_id} CLARIFAI_API_BASE={context.base_url} CLARIFAI_PAT={context.pat} '
  cmd += ' '.join([f'{k}={v}' for k, v in context.env.items()])
  cmd += f' {script}'
  os.system(cmd)

# Import the CLI commands to register them
load_command_modules()


def main():
  cli()
