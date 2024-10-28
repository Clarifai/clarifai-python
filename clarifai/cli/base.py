import os

import click

from ..utils.cli import dump_yaml, from_yaml, load_command_modules, set_base_url


@click.group()
@click.pass_context
def cli(ctx):
  """Clarifai CLI"""
  ctx.ensure_object(dict)
  config_path = 'config.yaml'
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
    click.echo("Loaded PAT from config file.")
  elif 'CLARIFAI_PAT' in os.environ:
    ctx.obj['pat'] = os.environ["CLARIFAI_PAT"]
    click.echo("Loaded PAT from environment variable.")
  else:
    _pat = click.prompt(
        "Get your PAT from https://clarifai.com/settings/security and pass it here", type=str)
    os.environ["CLARIFAI_PAT"] = _pat
    ctx.obj['pat'] = _pat
    click.echo("PAT saved successfully.")

  if user_id:
    ctx.obj['user_id'] = user_id
    os.environ["CLARIFAI_USER_ID"] = ctx.obj['user_id']
  elif 'user_id' in ctx.obj or 'CLARIFAI_USER_ID' in os.environ:
    ctx.obj['user_id'] = ctx.obj.get('user_id', os.environ["CLARIFAI_USER_ID"])
    os.environ["CLARIFAI_USER_ID"] = ctx.obj['user_id']

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

  dump_yaml(ctx.obj, 'config.yaml')


# Import the CLI commands to register them
load_command_modules()

if __name__ == '__main__':
  cli()
