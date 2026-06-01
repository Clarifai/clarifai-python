"""Orchestrate a full local pipeline run.

Ties together preflight checks, image building, spec adaptation,
secret creation, workflow submission, and log streaming.
"""

import os
from typing import Dict, Optional

import yaml

from clarifai.runners.pipelines.local.image_loader import build_and_load_all_steps
from clarifai.runners.pipelines.local.log_streamer import stream_workflow_logs
from clarifai.runners.pipelines.local.preflight import run_all_checks
from clarifai.runners.pipelines.local.secrets import (
    SECRET_NAME,
    build_env_vars,
    create_env_secret,
)
from clarifai.runners.pipelines.local.spec_adapter import adapt_spec_for_local, load_argo_spec_from_config
from clarifai.runners.pipelines.local.submitter import submit_workflow
from clarifai.utils.logging import logger


def run_local_pipeline(
    pipeline_dir: str,
    namespace: str = 'clarifai-local',
    pat: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: int = 3600,
    poll_interval: int = 3,
):
    """Execute a pipeline locally on the current K8s cluster.

    Args:
        pipeline_dir: Path to the pipeline directory containing config.yaml and step subdirectories.
        namespace: K8s namespace to use.
        pat: Clarifai PAT for step env vars. Falls back to CLARIFAI_PAT env var.
        api_base: Clarifai API base URL.
        timeout: Max wait time in seconds.
        poll_interval: Seconds between status polls.

    Returns:
        Final workflow phase string (e.g. 'Succeeded', 'Failed').
    """
    pipeline_dir = os.path.abspath(pipeline_dir)

    # Load pipeline config
    config_path = _resolve_config(pipeline_dir)
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    pipeline_config = config_data.get('pipeline', config_data)
    step_directories = pipeline_config.get('step_directories', [])
    user_id = pipeline_config.get('user_id', '')
    app_id = pipeline_config.get('app_id', '')

    if not step_directories:
        raise ValueError('No step_directories found in pipeline config.')

    # 1. Preflight checks
    logger.info('Running preflight checks ...')
    cluster_type = run_all_checks(namespace=namespace)

    # 2. Build and load step images
    logger.info('Building and loading step images ...')
    step_images = build_and_load_all_steps(pipeline_dir, step_directories, cluster_type)

    # 3. Parse Argo spec from config
    argo_spec = load_argo_spec_from_config(config_data)
    if not argo_spec:
        raise ValueError('No orchestration_spec.argo_orchestration_spec found in pipeline config.')

    # 4. Create K8s secret with env vars
    env_vars = build_env_vars(pat=pat, api_base=api_base, user_id=user_id, app_id=app_id)
    secret_name = None
    if env_vars:
        secret_name = create_env_secret(namespace, env_vars)

    # 5. Adapt spec for local execution
    adapted_spec = adapt_spec_for_local(
        argo_spec,
        step_images,
        namespace=namespace,
        env_secret_name=secret_name,
    )

    # 6. Submit workflow
    wf_name = submit_workflow(adapted_spec, namespace=namespace)

    # 7. Stream logs and wait for completion
    phase = stream_workflow_logs(
        wf_name,
        namespace=namespace,
        poll_interval=poll_interval,
        timeout=timeout,
    )

    if phase == 'Succeeded':
        logger.info(f'Pipeline completed successfully!')
    else:
        logger.error(f'Pipeline finished with phase: {phase}')

    return phase


def _resolve_config(pipeline_dir):
    """Find the config file in a pipeline directory."""
    for name in ('config-lock.yaml', 'config.yaml'):
        path = os.path.join(pipeline_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f'No config.yaml or config-lock.yaml found in {pipeline_dir}'
    )
