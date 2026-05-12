"""Create K8s Secrets for pipeline step environment variables."""

import json
import subprocess

from clarifai.utils.logging import logger

SECRET_NAME = 'clarifai-local-pipeline-env'


def create_env_secret(namespace, env_vars, secret_name=SECRET_NAME):
    """Create or update a K8s Secret with the given environment variables.

    Args:
        namespace: K8s namespace.
        env_vars: Dict of env var name -> value.
        secret_name: Name of the K8s Secret.
    """
    # Delete existing secret if present (idempotent)
    subprocess.run(
        ['kubectl', 'delete', 'secret', secret_name, '-n', namespace, '--ignore-not-found'],
        capture_output=True,
        text=True,
    )

    # Build --from-literal args
    cmd = ['kubectl', 'create', 'secret', 'generic', secret_name, '-n', namespace]
    for key, value in env_vars.items():
        cmd.append(f'--from-literal={key}={value}')

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'Failed to create secret {secret_name}: {result.stderr}')

    logger.info(f'Created K8s secret {secret_name} in namespace {namespace}.')
    return secret_name


def build_env_vars(pat=None, api_base=None, user_id=None, app_id=None):
    """Build the standard Clarifai env vars dict for pipeline steps."""
    import os

    env = {}

    pat = pat or os.environ.get('CLARIFAI_PAT', '')
    if pat:
        env['CLARIFAI_PAT'] = pat

    api_base = api_base or os.environ.get('CLARIFAI_API_BASE', 'https://api.clarifai.com')
    env['CLARIFAI_API_BASE'] = api_base

    if user_id:
        env['CLARIFAI_USER_ID'] = user_id
    if app_id:
        env['CLARIFAI_APP_ID'] = app_id

    return env
