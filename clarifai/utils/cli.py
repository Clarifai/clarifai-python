import importlib
import os
import pkgutil

import click
import yaml

from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text


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


def set_base_url(env):
  environments = {
      'prod': 'https://api.clarifai.com',
      'staging': 'https://api-staging.clarifai.com',
      'dev': 'https://api-dev.clarifai.com'
  }

  if env in environments:
    return environments[env]
  else:
    raise ValueError("Invalid environment. Please choose from 'prod', 'staging', 'dev'.")


# Dynamically find and import all command modules from the cli directory
def load_command_modules():
  package_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cli')

  for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    if module_name != 'base':  # Skip the base.py file itself
      importlib.import_module(f'clarifai.cli.{module_name}')


def display_co_resources(response, resource_type):
  """Display compute orchestration resources listing results using rich."""

  console = Console()
  panel = Panel(
      Text(f"List of {resource_type}s", justify="center"),
      title="",
      style=Style(color="blue", bold=True),
      border_style="green",
      width=60)
  console.print(panel)
  for indx, item in enumerate(list(response)):
    panel = Panel(
        "\n".join([f"{'ID'}: {item.id}", f"{'Description'}: {item.description}"]),
        title=f"{resource_type} {(indx + 1)}",
        border_style="green",
        width=60)
    console.print(panel)
