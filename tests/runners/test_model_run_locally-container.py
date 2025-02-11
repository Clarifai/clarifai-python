import os
import shutil
import subprocess
import sys

import pytest
from clarifai.runners.models.model_run_locally import ModelRunLocally

CLARIFAI_USER_ID = os.environ["CLARIFAI_USER_ID"]
CLARIFAI_PAT = os.environ["CLARIFAI_PAT"]


@pytest.fixture
def model_run_locally(dummy_models_path):
  """
  Fixture that instantiates the ModelRunLocally class
  with the dummy model_path that already exists.
  """
  return ModelRunLocally(dummy_models_path)


@pytest.fixture
def hf_model_run_locally(dummy_hf_models_path):
  """
  Fixture that instantiates the ModelRunLocally class
  with the dummy model_path that already exists.
  """
  return ModelRunLocally(dummy_hf_models_path)


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker not installed or not in PATH.")
@pytest.mark.skipif(
    sys.platform not in ["linux", "darwin"],
    reason="Test only runs on Linux and macOS because base image only supports those platforms.")
def test_docker_build_and_test_container(model_run_locally):
  """
  Test building a Docker image and running a container test using the dummy model.
  This test will be skipped if Docker is not installed.
  """

  # Test if Docker is installed
  assert model_run_locally.is_docker_installed(), "Docker not installed, skipping."

  # Build or re-build the Docker image
  model_run_locally.builder.create_dockerfile()
  image_tag = model_run_locally._docker_hash()
  image_name = f"{model_run_locally.config['model']['id']}:{image_tag}"

  if not model_run_locally.docker_image_exists(image_name):
    model_run_locally.build_docker_image(image_name=image_name)

  # Run tests inside the container
  try:
    model_run_locally.test_model_container(
        image_name=image_name,
        container_name="test_clarifai_model_container",
        env_vars={
            'CLARIFAI_PAT': CLARIFAI_PAT,
            'CLARIFAI_API_BASE': os.environ.get('CLARIFAI_API_BASE', 'https://api.clarifai.com')
        })
  except subprocess.CalledProcessError:
    pytest.fail("Failed to test the model inside the docker container.")
  finally:
    # Clean up the container if it still exists
    if model_run_locally.container_exists("test_clarifai_model_container"):
      model_run_locally.stop_docker_container("test_clarifai_model_container")
      model_run_locally.remove_docker_container("test_clarifai_model_container")

    # Remove the image
    model_run_locally.remove_docker_image(image_name)


# Skip the test if Docker is not installed or if the platform is not Linux/macOS
@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker not installed or not in PATH.")
@pytest.mark.skipif(
    sys.platform not in ["linux", "darwin"],
    reason="Test only runs on Linux and macOS because base image only supports those platforms.")
def test_hf_docker_build_and_test_container(hf_model_run_locally):
  """
  Test building a Docker image and running a container test using the dummy model.
  This test will be skipped if Docker is not installed.
  """

  # Download the checkpoints for the model
  hf_model_run_locally.builder.download_checkpoints()

  # Test if Docker is installed
  assert hf_model_run_locally.is_docker_installed(), "Docker not installed, skipping."

  # Build or re-build the Docker image
  hf_model_run_locally.builder.create_dockerfile()
  image_tag = hf_model_run_locally._docker_hash()
  image_name = f"{hf_model_run_locally.config['model']['id']}:{image_tag}"

  if not hf_model_run_locally.docker_image_exists(image_name):
    hf_model_run_locally.build_docker_image(image_name=image_name)

  # Run tests inside the container
  try:
    hf_model_run_locally.test_model_container(
        image_name=image_name,
        container_name="test_clarifai_model_container",
        env_vars={
            'CLARIFAI_PAT': CLARIFAI_PAT,
            'CLARIFAI_API_BASE': os.environ.get('CLARIFAI_API_BASE', 'https://api.clarifai.com')
        })
  except subprocess.CalledProcessError:
    pytest.fail("Failed to test the model inside the docker container.")
  finally:
    # Clean up the container if it still exists
    if hf_model_run_locally.container_exists("test_clarifai_model_container"):
      hf_model_run_locally.stop_docker_container("test_clarifai_model_container")
      hf_model_run_locally.remove_docker_container("test_clarifai_model_container")

    # Remove the image
    hf_model_run_locally.remove_docker_image(image_name)
