import click
import sys
import os
import typing as t
import yaml

from collections import defaultdict
from dataclasses import dataclass, field
from ..utils.cli import dump_yaml, from_yaml, load_command_modules, TableFormatter


class AliasedGroup(click.Group):

  def __init__(self,
               name: t.Optional[str] = None,
               commands: t.Optional[t.Union[t.MutableMapping[str, click.Command],
                                            t.Sequence[click.Command]]] = None,
               **attrs: t.Any) -> None:
    super().__init__(name, commands, **attrs)
    self.alias_map = {}
    self.command_to_aliases = defaultdict(list)

  def add_alias(self, cmd: click.Command, alias: str) -> None:
    self.alias_map[alias] = cmd
    if alias != cmd.name:
      self.command_to_aliases[cmd].append(alias)

  def command(self,
              aliases=None,
              *args,
              **kwargs) -> t.Callable[[t.Callable[..., t.Any]], click.Command]:
    cmd_decorator = super().command(*args, **kwargs)
    if aliases is None:
      aliases = []

    def aliased_decorator(f):
      cmd = cmd_decorator(f)
      if cmd.name:
        self.add_alias(cmd, cmd.name)
      for alias in aliases:
        self.add_alias(cmd, alias)
      return cmd

    f = None
    if args and callable(args[0]):
      (f,) = args
    if f is not None:
      return aliased_decorator(f)
    return aliased_decorator

  def group(self, aliases=None, *args, **kwargs) -> t.Callable[[t.Callable[..., t.Any]], click.Group]:
    cmd_decorator = super().group(*args, **kwargs)
    if aliases is None:
      aliases = []

    def aliased_decorator(f):
      cmd = cmd_decorator(f)
      if cmd.name:
        self.add_alias(cmd, cmd.name)
      for alias in aliases:
        self.add_alias(cmd, alias)
      return cmd

    f = None
    if args and callable(args[0]):
      (f,) = args
    if f is not None:
      return aliased_decorator(f)
    return aliased_decorator

  def get_command(self, ctx: click.Context, cmd_name: str) -> t.Optional[click.Command]:
    rv = click.Group.get_command(self, ctx, cmd_name)
    if rv is not None:
      return rv
    return self.alias_map.get(cmd_name)

  def format_commands(self, ctx, formatter):
    sub_commands = self.list_commands(ctx)

    rows = []
    for sub_command in sub_commands:
      cmd = self.get_command(ctx, sub_command)
      if cmd is None or getattr(cmd, 'hidden', False):
        continue
      if cmd in self.command_to_aliases:
        aliases = ', '.join(self.command_to_aliases[cmd])
        sub_command = f'{sub_command} ({aliases})'
      cmd_help = cmd.help
      rows.append((sub_command, cmd_help))

    if rows:
      with formatter.section("Commands"):
        formatter.write_dl(rows)

@dataclass
class Context():
  name: str
  user_id: str
  base_url: str
  access_token: dict = field(default_factory=lambda: dict(type='env', value='CLARIFAI_PAT'))
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

  def to_serializable_dict(self):
    return {
      'name': self.name,
      'user_id': self.user_id,
      'base_url': self.base_url,
      'access_token': self.access_token
    }


@dataclass
class Config():
  current_context: str
  filename: str
  contexts: dict[str, Context] = field(default_factory=dict)

  def __post_init__(self):
    self.contexts = {k: Context(name=k, **v,) for k, v in self.contexts.items()}

  @classmethod
  def from_yaml(cls, filename: str = None):
    with open(filename, 'r') as f:
      cfg = yaml.safe_load(f)
    return cls(**cfg, filename=filename)

  def to_dict(self):
    return {
              'current_context': self.current_context,
              'contexts': {
                  k: dict(
                      user_id=v.user_id,
                      base_url=v.base_url,
                      access_token=v.access_token,
                  ) for k, v in self.contexts.items()
              }
          }

  def to_yaml(self, filename: str = None):
    if filename is None:
      filename = self.filename
    dir = os.path.dirname(filename)
    if len(dir):
      os.makedirs(dir, exist_ok=True)
    with open(filename, 'w') as f:
      yaml.safe_dump(
          self.to_dict(), f)

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

@cli.group(['cfg'])
def config():
  """Manage CLI configuration"""
  pass

@config.command()
@click.pass_context
def edit(ctx):
  os.system(f'{os.environ.get("EDITOR", "vi")} {ctx.obj.filename}')

@config.command()
@click.option('-o', '--output-format', default='name', type=click.Choice(['name','json', 'yaml']))
@click.pass_context
def current_context(ctx, output_format):
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
  if output_format == 'wide':
    formatter = TableFormatter(
        custom_columns={
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


@config.command()
@click.argument('context-name', type=str)
@click.pass_context
def use(ctx, context_name):
  if context_name not in ctx.obj.contexts:
    raise click.UsageError('Context not found')
  ctx.obj.current_context = context_name
  ctx.obj.to_yaml()
  print(f'Set {context_name} as the current context')

@config.command()
@click.option('-o', '--output-format', default='yaml', type=click.Choice(['yaml','json']))
@click.pass_obj
def dump(ctx_obj, output_format):
  if output_format == 'yaml':
    yaml.safe_dump(ctx_obj.to_dict(), sys.stdout)
  else:
    json.dump(ctx_obj.to_dict(), sys.stdout, indent=2)

@cli.command()
@click.argument('api_url')
@click.option('--user_id', required=False, help='User ID')
@click.pass_context
def login(ctx, api_url, user_id):
  """Login command to set PAT and other configurations."""

  context = Context(
    name=input('context name: '),
    base_url=api_url,
    user_id=user_id if user_id is not None else input('user id: '),
    access_token=dict(
      type=input('access token type: '),
      value=input('access token value: '),
    ),
  )

  ctx.obj.contexts[context.name] = context
  ctx.obj.current_context = context.name

  ctx.obj.to_yaml()

# Import the CLI commands to register them
load_command_modules()

if __name__ == '__main__':
  cli()
