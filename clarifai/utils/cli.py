import importlib
import os
import pkgutil
import sys
import typing as t
from collections import defaultdict
from typing import OrderedDict

import click
import yaml
from tabulate import tabulate

from clarifai.utils.logging import logger


def from_yaml(filename: str):
  try:
    with open(filename, 'r') as f:
      return yaml.safe_load(f)
  except yaml.YAMLError as e:
    click.echo(f"Error reading YAML file: {e}", err=True)
    return {}


def dump_yaml(data, filename: str):
  try:
    with open(filename, 'w') as f:
      yaml.dump(data, f)
  except Exception as e:
    click.echo(f"Error writing YAML file: {e}", err=True)


# Dynamically find and import all command modules from the cli directory
def load_command_modules():
  package_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cli')

  for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    if module_name not in ['base', '__main__']:  # Skip the base.py and __main__ file itself
      importlib.import_module(f'clarifai.cli.{module_name}')


def display_co_resources(response,
                         custom_columns={
                             'ID': lambda c: c.id,
                             'USER_ID': lambda c: c.user_id,
                             'DESCRIPTION': lambda c: c.description,
                         }):
  """Display compute orchestration resources listing results using rich."""

  formatter = TableFormatter(custom_columns)
  print(formatter.format(list(response), fmt="plain"))


class TableFormatter:

  def __init__(self, custom_columns: OrderedDict):
    """
        Initializes the TableFormatter with column headers and custom column mappings.

        :param headers: List of column headers for the table.
        """
    self.custom_columns = custom_columns

  def format(self, objects, fmt='plain'):
    """
        Formats a list of objects into a table with custom columns.

        :param objects: List of objects to format into a table.
        :return: A string representing the table.
        """
    # Prepare the rows by applying the custom column functions to each object
    rows = []
    for obj in objects:
      #   row = [self.custom_columns[header](obj) for header in self.headers]
      row = [f(obj) for f in self.custom_columns.values()]
      rows.append(row)

    # Create the table
    table = tabulate(rows, headers=self.custom_columns.keys(), tablefmt=fmt)
    return table


class AliasedGroup(click.Group):

  def __init__(self,
               name: t.Optional[str] = None,
               commands: t.Optional[t.Union[t.MutableMapping[str, click.Command], t.Sequence[
                   click.Command]]] = None,
               **attrs: t.Any) -> None:
    super().__init__(name, commands, **attrs)
    self.alias_map = {}
    self.command_to_aliases = defaultdict(list)

  def add_alias(self, cmd: click.Command, alias: str) -> None:
    self.alias_map[alias] = cmd
    if alias != cmd.name:
      self.command_to_aliases[cmd].append(alias)

  def command(self, aliases=None, *args,
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

  def group(self, aliases=None, *args,
            **kwargs) -> t.Callable[[t.Callable[..., t.Any]], click.Group]:
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


def validate_context(ctx):
  if ctx.obj == {}:
    logger.error("CLI config file missing. Run `clarifai login` to set up the CLI config.")
    sys.exit(1)
