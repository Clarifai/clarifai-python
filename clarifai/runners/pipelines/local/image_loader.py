"""Build Docker images for pipeline steps and load them into the local K8s cluster."""

import os
import subprocess

from clarifai.runners.pipeline_steps.pipeline_run_locally import PipelineStepRunLocally
from clarifai.utils.logging import logger


def _image_name_for_step(step_id, step_path):
    """Generate a deterministic local image name for a pipeline step."""
    manager = PipelineStepRunLocally(step_path)
    # Ensure Dockerfile exists before computing hash
    manager.builder.create_dockerfile()
    tag = manager._docker_hash()
    return f'clarifai-local/{step_id}:{tag}'


def build_step_image(step_path):
    """Build a Docker image for a single pipeline step.

    Returns the image name (tag).
    """
    manager = PipelineStepRunLocally(step_path)
    step_id = manager.config['pipeline_step']['id'].lower()
    image_name = _image_name_for_step(step_id, step_path)

    if manager.docker_image_exists(image_name):
        logger.info(f'Image {image_name} already exists, skipping build.')
    else:
        logger.info(f'Building Docker image {image_name} ...')
        manager.build_docker_image(image_name=image_name)

    return image_name


def load_image_into_cluster(image_name, cluster_type):
    """Load a Docker image into the local K8s cluster."""
    logger.info(f'Loading image {image_name} into {cluster_type} cluster ...')

    if cluster_type == 'minikube':
        # Use docker save | minikube image load - for reliability
        result = subprocess.run(
            f'docker save {image_name} | minikube image load --daemon=false -',
            shell=True,
            capture_output=True,
            text=True,
        )
    elif cluster_type == 'k3d':
        result = subprocess.run(
            ['k3d', 'image', 'import', image_name],
            capture_output=True,
            text=True,
        )
    elif cluster_type == 'kind':
        result = subprocess.run(
            ['kind', 'load', 'docker-image', image_name],
            capture_output=True,
            text=True,
        )
    else:
        logger.warning(
            f'Unknown cluster type {cluster_type}. Assuming image is accessible via Docker daemon.'
        )
        return

    if result.returncode != 0:
        raise RuntimeError(
            f'Failed to load image {image_name} into {cluster_type}: {result.stderr}'
        )
    logger.info(f'Loaded {image_name} into cluster.')


def build_and_load_all_steps(pipeline_dir, step_directories, cluster_type):
    """Build and load images for all pipeline steps.

    Returns a dict mapping step_id -> image_name.
    """
    step_images = {}
    for step_dir_name in step_directories:
        step_path = os.path.join(pipeline_dir, step_dir_name)
        if not os.path.isdir(step_path):
            raise FileNotFoundError(f'Step directory not found: {step_path}')

        manager = PipelineStepRunLocally(step_path)
        step_id = manager.config['pipeline_step']['id']

        image_name = build_step_image(step_path)
        load_image_into_cluster(image_name, cluster_type)
        step_images[step_id] = image_name

    return step_images
