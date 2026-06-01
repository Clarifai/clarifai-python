"""Adapt an Argo Workflow spec for local execution.

Converts remote templateRef-based specs into self-contained inline templates.
"""

import copy
import os
from typing import Any, Dict, List, Optional

import yaml

from clarifai.utils.logging import logger


def _build_inline_template(step_id, image_name, input_params):
    """Build an inline Argo template for a pipeline step.

    The step runs ``python /home/nonroot/main/1/pipeline_step.py`` with
    argparse-style arguments derived from input parameters.
    """
    args = []
    for param in input_params:
        name = param['name']
        args.extend([f'--{name}', f'{{{{inputs.parameters.{name}}}}}'])

    template = {
        'name': step_id,
        'inputs': {
            'parameters': [{'name': p['name']} for p in input_params],
        },
        'outputs': {
            'parameters': [
                {
                    'name': 'result',
                    'valueFrom': {'path': '/tmp/result'},
                    'globalName': f'{step_id}-result',
                }
            ],
        },
        'container': {
            'image': image_name,
            'imagePullPolicy': 'Never',
            'command': ['python', '/home/nonroot/main/1/pipeline_step.py'],
            'args': args,
        },
    }

    if not input_params:
        del template['inputs']
        template['container']['args'] = []

    return template


def _extract_step_id_from_template_ref(template_ref_name):
    """Extract step_id from a templateRef name like 'users/X/apps/Y/pipeline_steps/Z'."""
    parts = template_ref_name.split('/')
    # Pattern: users/{user}/apps/{app}/pipeline_steps/{step_id}[/versions/{ver}]
    try:
        idx = parts.index('pipeline_steps')
        return parts[idx + 1]
    except (ValueError, IndexError):
        return template_ref_name


def _collect_step_params_from_spec(argo_spec):
    """Collect parameter names per step from the workflow spec's step arguments."""
    step_params = {}
    templates = argo_spec.get('spec', {}).get('templates', [])
    for template in templates:
        for step_group in template.get('steps', []):
            for step_entry in step_group:
                ref = step_entry.get('templateRef', {})
                ref_name = ref.get('name', '')
                step_id = _extract_step_id_from_template_ref(ref_name)

                params = []
                for p in step_entry.get('arguments', {}).get('parameters', []):
                    params.append({'name': p['name']})
                step_params[step_id] = params
    return step_params


def adapt_spec_for_local(
    argo_spec,
    step_images,
    namespace='clarifai-local',
    env_secret_name=None,
):
    """Transform a remote Argo Workflow spec for local execution.

    - Replaces ``templateRef`` entries with references to inline templates
    - Adds inline container templates for each step
    - Sets ``imagePullPolicy: Never``
    - Removes cloud-specific affinity/tolerations
    - Adds metadata (name, namespace, labels)

    Args:
        argo_spec: The parsed Argo Workflow spec dict.
        step_images: Dict mapping step_id -> local Docker image name.
        namespace: K8s namespace to run in.
        env_secret_name: Optional K8s Secret name to inject as envFrom.

    Returns:
        The adapted workflow spec dict, ready for K8s submission.
    """
    spec = copy.deepcopy(argo_spec)

    # Collect parameter info from step arguments in the spec
    step_params = _collect_step_params_from_spec(spec)

    # Build inline templates for each step
    inline_templates = []
    for step_id, image_name in step_images.items():
        params = step_params.get(step_id, [])
        tmpl = _build_inline_template(step_id, image_name, params)

        # Inject env secret if provided
        if env_secret_name:
            tmpl['container']['envFrom'] = [
                {'secretRef': {'name': env_secret_name}},
            ]

        inline_templates.append(tmpl)

    # Replace templateRef with template in step entries
    for template in spec.get('spec', {}).get('templates', []):
        for step_group in template.get('steps', []):
            for step_entry in step_group:
                if 'templateRef' in step_entry:
                    ref_name = step_entry['templateRef'].get('name', '')
                    step_id = _extract_step_id_from_template_ref(ref_name)
                    del step_entry['templateRef']
                    step_entry['template'] = step_id

    # Add inline templates
    spec['spec']['templates'].extend(inline_templates)

    # Remove cloud-specific scheduling constraints
    for key in ('affinity', 'tolerations', 'nodeSelector'):
        spec['spec'].pop(key, None)

    # Set metadata
    entrypoint = spec['spec'].get('entrypoint', 'pipeline')
    wf_name = f'local-{entrypoint}'[:63]  # K8s name limit
    spec.setdefault('metadata', {})
    spec['metadata']['name'] = wf_name
    spec['metadata']['namespace'] = namespace
    spec['metadata'].setdefault('labels', {})
    spec['metadata']['labels']['clarifai.com/local-run'] = 'true'

    # Ensure generateName isn't set (conflicts with name)
    spec['metadata'].pop('generateName', None)

    # Set service account for Argo
    spec['spec'].setdefault('serviceAccountName', 'argo')

    return spec


def load_argo_spec_from_config(config_data):
    """Extract and parse the Argo orchestration spec from a pipeline config dict."""
    pipeline_config = config_data.get('pipeline', config_data)
    orch_spec = pipeline_config.get('orchestration_spec', {})
    argo_spec_raw = orch_spec.get('argo_orchestration_spec', '')

    if isinstance(argo_spec_raw, str):
        return yaml.safe_load(argo_spec_raw)
    return argo_spec_raw
