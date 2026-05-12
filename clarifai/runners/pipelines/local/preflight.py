"""Preflight checks for local pipeline execution.

Validates that the local K8s cluster and Argo Workflows are ready.
"""

import shutil
import subprocess

from clarifai.utils.logging import logger

ARGO_WORKFLOW_CRD = 'workflows.argoproj.io'


def check_docker():
    """Verify Docker is installed and running."""
    if not shutil.which('docker'):
        raise EnvironmentError('Docker is not installed or not on PATH.')
    result = subprocess.run(
        ['docker', 'info'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EnvironmentError('Docker daemon is not running. Please start Docker Desktop.')
    logger.info('Docker is running.')


def check_kubectl():
    """Verify kubectl is available and a cluster is reachable."""
    if not shutil.which('kubectl'):
        raise EnvironmentError('kubectl is not installed or not on PATH.')
    result = subprocess.run(
        ['kubectl', 'cluster-info'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EnvironmentError(
            'Cannot connect to Kubernetes cluster. Is minikube/k3s running?\n'
            f'  stderr: {result.stderr.strip()}'
        )
    logger.info('Kubernetes cluster is reachable.')


def check_argo_crds():
    """Verify Argo Workflow CRDs are registered in the cluster."""
    result = subprocess.run(
        ['kubectl', 'get', 'crd', ARGO_WORKFLOW_CRD],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EnvironmentError(
            'Argo Workflows CRDs not found in cluster.\n'
            'Install Argo Workflows first:\n'
            '  kubectl create namespace argo\n'
            '  kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.12/quick-start-minimal.yaml'
        )
    logger.info('Argo Workflows CRDs found.')


def detect_cluster_type():
    """Detect the local K8s cluster type for image loading strategy.

    Returns one of: 'minikube', 'k3d', 'kind', or 'generic'.
    """
    result = subprocess.run(
        ['kubectl', 'config', 'current-context'],
        capture_output=True,
        text=True,
    )
    context = result.stdout.strip() if result.returncode == 0 else ''

    if 'minikube' in context:
        return 'minikube'
    if 'k3d' in context:
        return 'k3d'
    if 'kind' in context:
        return 'kind'
    return 'generic'


def ensure_namespace(namespace):
    """Create the namespace if it doesn't exist."""
    result = subprocess.run(
        ['kubectl', 'get', 'namespace', namespace],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        subprocess.run(
            ['kubectl', 'create', 'namespace', namespace],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f'Created namespace {namespace}.')


def run_all_checks(namespace='clarifai-local'):
    """Run all preflight checks. Returns the detected cluster type."""
    check_docker()
    check_kubectl()
    check_argo_crds()
    ensure_namespace(namespace)
    cluster_type = detect_cluster_type()
    logger.info(f'Detected cluster type: {cluster_type}')
    return cluster_type
