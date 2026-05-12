"""Submit an Argo Workflow to the local K8s cluster."""

import json
import subprocess

from clarifai.utils.logging import logger


def submit_workflow(workflow_spec, namespace='clarifai-local'):
    """Submit an Argo Workflow CR to the local K8s cluster via kubectl.

    Args:
        workflow_spec: The full Argo Workflow dict (with apiVersion, kind, metadata, spec).
        namespace: K8s namespace.

    Returns:
        The workflow name.
    """
    spec_json = json.dumps(workflow_spec)

    # Delete existing workflow with the same name if present (for re-runs)
    wf_name = workflow_spec.get('metadata', {}).get('name', '')
    if wf_name:
        subprocess.run(
            ['kubectl', 'delete', 'workflow', wf_name, '-n', namespace, '--ignore-not-found'],
            capture_output=True,
            text=True,
        )

    result = subprocess.run(
        ['kubectl', 'apply', '-n', namespace, '-f', '-'],
        input=spec_json,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'Failed to submit Argo Workflow: {result.stderr}')

    logger.info(f'Submitted Argo Workflow: {wf_name}')
    return wf_name
