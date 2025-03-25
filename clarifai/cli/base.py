import os

import click

from clarifai.utils.cli import dump_yaml, from_yaml, load_command_modules, set_base_url
from clarifai.utils.logging import logger


class CustomMultiGroup(click.Group):

  def group(self, *args, **kwargs):
    """Behaves the same as `click.Group.group()` except if passed
        a list of names, all after the first will be aliases for the first.
        """

    def decorator(f):
      aliased_group = []
      if isinstance(args[0], list):
        # we have a list so create group aliases
        _args = [args[0][0]] + list(args[1:])
        for alias in args[0][1:]:
          grp = super(CustomMultiGroup, self).group(alias, *args[1:], **kwargs)(f)
          grp.short_help = "Alias for '{}'".format(_args[0])
          aliased_group.append(grp)
      else:
        _args = args

      # create the main group
      grp = super(CustomMultiGroup, self).group(*_args, **kwargs)(f)

      # for all of the aliased groups, share the main group commands
      for aliased in aliased_group:
        aliased.commands = grp.commands

      return grp

    return decorator


@click.group(cls=CustomMultiGroup)
@click.pass_context
def cli(ctx):
  """Clarifai CLI"""
  ctx.ensure_object(dict)
  config_path = f"{os.getenv('HOME')}/.clarifai/config.yaml"
  if os.path.exists(config_path):
    ctx.obj = from_yaml(config_path)
  else:
    ctx.obj = {}


@cli.command()
@click.option('--config', type=click.Path(), required=False, help='Path to the config file')
@click.option(
    '-e',
    '--env',
    required=False,
    help='Environment to use, choose from prod, staging and dev',
    type=click.Choice(['prod', 'staging', 'dev']))
@click.option('--user_id', required=False, help='User ID')
@click.pass_context
def login(ctx, config, env, user_id):
  """Login command to set PAT and other configurations."""

  if config and os.path.exists(config):
    ctx.obj = from_yaml(config)

  if 'pat' in ctx.obj:
    os.environ["CLARIFAI_PAT"] = ctx.obj['pat']
    logger.info("Loaded PAT from config file.")
  elif 'CLARIFAI_PAT' in os.environ:
    ctx.obj['pat'] = os.environ["CLARIFAI_PAT"]
    logger.info("Loaded PAT from environment variable.")
  else:
    _pat = click.prompt(
        "Get your PAT from https://clarifai.com/settings/security and pass it here", type=str)
    os.environ["CLARIFAI_PAT"] = _pat
    ctx.obj['pat'] = _pat
    logger.info("PAT saved successfully.")

  if user_id:
    ctx.obj['user_id'] = user_id
    os.environ["CLARIFAI_USER_ID"] = ctx.obj['user_id']
  elif 'user_id' in ctx.obj:
    ctx.obj['user_id'] = ctx.obj.get('user_id', "")
    os.environ["CLARIFAI_USER_ID"] = ctx.obj['user_id']
  elif 'CLARIFAI_USER_ID' in os.environ:
    ctx.obj['user_id'] = os.environ["CLARIFAI_USER_ID"]
  else:
    user_id = click.prompt("Pass the User ID here", type=str)
    os.environ["CLARIFAI_USER_ID"] = user_id
    ctx.obj['user_id'] = user_id
    logger.info("User ID saved successfully.")

  if env:
    ctx.obj['env'] = env
    ctx.obj['base_url'] = set_base_url(env)
    os.environ["CLARIFAI_API_BASE"] = ctx.obj['base_url']
  elif 'env' in ctx.obj:
    ctx.obj['env'] = ctx.obj.get('env', "prod")
    ctx.obj['base_url'] = set_base_url(ctx.obj['env'])
    os.environ["CLARIFAI_API_BASE"] = ctx.obj['base_url']
  elif 'CLARIFAI_API_BASE' in os.environ:
    ctx.obj['base_url'] = os.environ["CLARIFAI_API_BASE"]
  else:
    ctx.obj['env'] = 'prod'
    ctx.obj['base_url'] = set_base_url(ctx.obj['env'])
    os.environ["CLARIFAI_API_BASE"] = ctx.obj['base_url']
    logger.info("Base URL saved successfully.")

  config_path = f"{os.getenv('HOME')}/.clarifai/config.yaml"
  config_dir = os.path.dirname(config_path)
  if not os.path.exists(config_dir):
    os.makedirs(config_dir)

  dump_yaml(ctx.obj, config_path)


# Import the CLI commands to register them
load_command_modules()


def main():
  cli()
