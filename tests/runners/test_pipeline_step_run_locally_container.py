import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from clarifai.runners.pipeline_steps.pipeline_run_locally import PipelineStepRunLocally


@pytest.fixture
def dummy_pipeline_step_path(tmp_path):
    """Copy the dummy_pipeline_step folder to a temp directory."""
    tests_dir = Path(__file__).parent.resolve()
    original_path = tests_dir / "dummy_pipeline_step"
    if not original_path.exists():
        raise FileNotFoundError(f"Could not find dummy_pipeline_step at {original_path}.")
    target_folder = tmp_path / "dummy_pipeline_step"
    shutil.copytree(original_path, target_folder)
    return str(target_folder)


@pytest.fixture
def pipeline_step_run_locally(dummy_pipeline_step_path):
    """Instantiate PipelineStepRunLocally with the dummy pipeline step."""
    return PipelineStepRunLocally(dummy_pipeline_step_path)


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker not installed or not in PATH.")
@pytest.mark.skipif(
    sys.platform not in ["linux", "darwin"],
    reason="Test only runs on Linux and macOS.",
)
def test_pipeline_step_docker_build_and_run(pipeline_step_run_locally):
    """Test building a Docker image and running a pipeline step in a container."""
    assert pipeline_step_run_locally.is_docker_installed(), "Docker not installed."

    pipeline_step_run_locally.builder.create_dockerfile()
    image_tag = pipeline_step_run_locally._docker_hash()
    step_id = pipeline_step_run_locally.config['pipeline_step']['id'].lower()
    image_name = f"{step_id}:{image_tag}"
    container_name = "test-pipeline-step-container"

    if not pipeline_step_run_locally.docker_image_exists(image_name):
        pipeline_step_run_locally.build_docker_image(image_name=image_name)

    try:
        pipeline_step_run_locally.run_pipeline_step_container(
            image_name=image_name,
            container_name=container_name,
        )
    except subprocess.CalledProcessError:
        pytest.fail("Failed to run pipeline step inside the docker container.")
    finally:
        if pipeline_step_run_locally.container_exists(container_name):
            pipeline_step_run_locally.stop_docker_container(container_name)
            pipeline_step_run_locally.remove_docker_container(container_name)
        pipeline_step_run_locally.remove_docker_image(image_name)
