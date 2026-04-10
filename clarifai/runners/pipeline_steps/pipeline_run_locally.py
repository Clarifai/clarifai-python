"""Run a pipeline step locally in a Docker container.

Reuses Docker infrastructure from ModelRunLocally for building and managing
containers, but runs pipeline_step.py instead of the model server.
"""

import os
import signal
import subprocess
import sys
import time

from clarifai.runners.models.model_run_locally import ModelRunLocally
from clarifai.runners.pipeline_steps.pipeline_step_builder import PipelineStepBuilder
from clarifai.utils.logging import logger


class PipelineStepRunLocally:
    """Run a single pipeline step locally in a Docker container.

    Reuses ModelRunLocally for Docker build/image/container management,
    but overrides the container command to run pipeline_step.py.
    """

    def __init__(self, step_path):
        self.step_path = os.path.abspath(step_path)
        self.builder = PipelineStepBuilder(self.step_path)
        self.config = self.builder.config

        # Create a ModelRunLocally instance for Docker utilities.
        # We bypass __init__ since it expects a ModelBuilder, and set the
        # attributes that the Docker methods rely on directly.
        self._docker = ModelRunLocally.__new__(ModelRunLocally)
        self._docker.model_path = self.step_path
        self._docker.requirements_file = os.path.join(self.step_path, "requirements.txt")

    # ── Delegated Docker utilities ──────────────────────────────────────

    def is_docker_installed(self):
        return self._docker.is_docker_installed()

    def docker_image_exists(self, image_name):
        return self._docker.docker_image_exists(image_name)

    def build_docker_image(self, image_name):
        return self._docker.build_docker_image(image_name=image_name)

    def container_exists(self, container_name):
        return self._docker.container_exists(container_name)

    def stop_docker_container(self, container_name):
        return self._docker.stop_docker_container(container_name)

    def remove_docker_container(self, container_name):
        return self._docker.remove_docker_container(container_name)

    def remove_docker_image(self, image_name):
        return self._docker.remove_docker_image(image_name)

    def _docker_hash(self):
        return self._docker._docker_hash()

    def _gpu_is_available(self):
        return self._docker._gpu_is_available()

    # ── Pipeline-step-specific container run ────────────────────────────

    def run_pipeline_step_container(
        self,
        image_name,
        container_name="clarifai-pipeline-step-container",
        env_vars=None,
        step_args=None,
    ):
        """Run pipeline_step.py inside a Docker container and wait for it to finish.

        Unlike ``ModelRunLocally.run_docker_container`` which starts a long-running
        server, this method executes the pipeline step script once and exits.

        Args:
            step_args: Optional list of arguments to pass to pipeline_step.py.
        """
        try:
            cmd = ["docker", "run", "--name", container_name, "--rm", "--network", "host"]

            if self._gpu_is_available():
                cmd.extend(["--gpus", "all"])

            # Mount step directory into container (same target as model serve)
            cmd.extend(
                [
                    "--mount",
                    f"type=bind,source={self.step_path},target=/home/nonroot/main",
                ]
            )

            if env_vars:
                for key, value in env_vars.items():
                    cmd.extend(["-e", f"{key}={value}"])

            cmd.extend(["-e", "PYTHONDONTWRITEBYTECODE=1"])

            # Override entrypoint to run pipeline_step.py directly
            cmd.extend(["--entrypoint", "python"])
            cmd.append(image_name)
            cmd.extend(["/home/nonroot/main/1/pipeline_step.py"])

            # Append any extra arguments for the pipeline step script
            if step_args:
                cmd.extend(step_args)

            logger.info(f"Running pipeline step in container '{container_name}'...")
            logger.info(f"Docker command: {' '.join(cmd)}")

            process = subprocess.Popen(cmd)

            # Graceful Ctrl+C handling
            original_sigint = signal.getsignal(signal.SIGINT)

            def signal_handler(sig, frame):
                logger.info(f"Stopping container '{container_name}'...")
                subprocess.run(["docker", "stop", container_name], check=False)
                process.terminate()
                signal.signal(signal.SIGINT, original_sigint)
                time.sleep(1)
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)

            process.wait()

            # Restore original handler
            signal.signal(signal.SIGINT, original_sigint)

            if process.returncode != 0:
                logger.error(f"Pipeline step failed with exit code {process.returncode}")
                sys.exit(process.returncode)

            logger.info("Pipeline step completed successfully!")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running pipeline step container: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error running pipeline step container: {e}")
            sys.exit(1)
