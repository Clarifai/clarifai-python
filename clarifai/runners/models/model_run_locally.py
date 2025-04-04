import hashlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import venv

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.utils.url_fetcher import ensure_urls_downloaded
from clarifai.utils.logging import logger


class ModelRunLocally:

  def __init__(self, model_path):
    self.model_path = model_path
    self.requirements_file = os.path.join(self.model_path, "requirements.txt")

    # ModelBuilder contains multiple useful methods to interact with the model
    self.builder = ModelBuilder(self.model_path, download_validation_only=True)
    self.config = self.builder.config

  def _requirements_hash(self):
    """Generate a hash of the requirements file."""
    with open(self.requirements_file, "r") as f:
      return hashlib.md5(f.read().encode('utf-8')).hexdigest()

  def _get_env_executable(self):
    """Get the python executable from the virtual environment."""
    # Depending on the platform, venv scripts are placed in either "Scripts" (Windows) or "bin" (Linux/Mac)
    if platform.system().lower().startswith("win"):
      scripts_folder = "Scripts"
      python_exe = "python.exe"
      pip_exe = "pip.exe"
    else:
      scripts_folder = "bin"
      python_exe = "python"
      pip_exe = "pip"

    self.python_executable = os.path.join(self.venv_dir, scripts_folder, python_exe)
    self.pip_executable = os.path.join(self.venv_dir, scripts_folder, pip_exe)

    return self.python_executable, self.pip_executable

  def create_temp_venv(self):
    """Create a temporary virtual environment."""
    requirements_hash = self._requirements_hash()

    temp_dir = os.path.join(tempfile.gettempdir(), str(requirements_hash))
    venv_dir = os.path.join(temp_dir, "venv")

    if os.path.exists(temp_dir):
      logger.info(f"Using previous virtual environment at {temp_dir}")
      use_existing_venv = True
    else:
      logger.info("Creating temporary virtual environment...")
      use_existing_venv = False
      venv.create(venv_dir, with_pip=True)
      logger.info(f"Created temporary virtual environment at {venv_dir}")

    self.venv_dir = venv_dir
    self.temp_dir = temp_dir
    self.python_executable, self.pip_executable = self._get_env_executable()

    return use_existing_venv

  def install_requirements(self):
    """Install the dependencies from requirements.txt and Clarifai."""
    _, pip_executable = self._get_env_executable()
    try:
      logger.info(
          f"Installing requirements from {self.requirements_file}... in the virtual environment {self.venv_dir}"
      )
      subprocess.check_call([pip_executable, "install", "-r", self.requirements_file])
      logger.info("Installing Clarifai package...")
      subprocess.check_call([pip_executable, "install", "clarifai"])
      logger.info("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
      logger.error(f"Error installing requirements: {e}")
      self.clean_up()
      sys.exit(1)

  def _build_request(self):
    """Create a mock inference request for testing the model."""

    model_version_proto = self.builder.get_model_version_proto()
    model_version_proto.id = "model_version"

    return service_pb2.PostModelOutputsRequest(
        model=resources_pb2.Model(model_version=model_version_proto),
        inputs=[
            resources_pb2.Input(data=resources_pb2.Data(
                text=resources_pb2.Text(raw="How many people live in new york?"),
                image=resources_pb2.Image(url="https://samples.clarifai.com/metro-north.jpg"),
                audio=resources_pb2.Audio(url="https://samples.clarifai.com/GoodMorning.wav"),
                video=resources_pb2.Video(url="https://samples.clarifai.com/beer.mp4"),
            ))
        ],
    )

  def _build_stream_request(self):
    request = self._build_request()
    ensure_urls_downloaded(request)
    for i in range(1):
      yield request

  def _run_model_inference(self, model):
    """Perform inference using the model."""
    request = self._build_request()
    stream_request = self._build_stream_request()

    ensure_urls_downloaded(request)
    predict_response = None
    generate_response = None
    stream_response = None
    try:
      predict_response = model.predict(request)
    except NotImplementedError:
      logger.info("Model does not implement predict() method.")
    except Exception as e:
      logger.error(f"Model Prediction failed: {e}")
      traceback.print_exc()
      predict_response = service_pb2.MultiOutputResponse(status=status_pb2.Status(
          code=status_code_pb2.MODEL_PREDICTION_FAILED,
          description="Prediction failed",
          details="",
          internal_details=str(e),
      ))

    if predict_response:
      if predict_response.outputs[0].status.code != status_code_pb2.SUCCESS:
        logger.error(f"Moddel Prediction failed: {predict_response}")
      else:
        logger.info(f"Model Prediction succeeded: {predict_response}")

    try:
      generate_response = model.generate(request)
    except NotImplementedError:
      logger.info("Model does not implement generate() method.")
    except Exception as e:
      logger.error(f"Model Generation failed: {e}")
      traceback.print_exc()
      generate_response = service_pb2.MultiOutputResponse(status=status_pb2.Status(
          code=status_code_pb2.MODEL_GENERATION_FAILED,
          description="Generation failed",
          details="",
          internal_details=str(e),
      ))

    if generate_response:
      generate_first_res = next(generate_response)
      if generate_first_res.outputs[0].status.code != status_code_pb2.SUCCESS:
        logger.error(f"Moddel Prediction failed: {generate_first_res}")
      else:
        logger.info(
            f"Model Prediction succeeded for generate and first response: {generate_first_res}")

    try:
      stream_response = model.stream(stream_request)
    except NotImplementedError:
      logger.info("Model does not implement stream() method.")
    except Exception as e:
      logger.error(f"Model Stream failed: {e}")
      traceback.print_exc()
      stream_response = service_pb2.MultiOutputResponse(status=status_pb2.Status(
          code=status_code_pb2.MODEL_STREAM_FAILED,
          description="Stream failed",
          details="",
          internal_details=str(e),
      ))

    if stream_response:
      stream_first_res = next(stream_response)
      if stream_first_res.outputs[0].status.code != status_code_pb2.SUCCESS:
        logger.error(f"Moddel Prediction failed: {stream_first_res}")
      else:
        logger.info(
            f"Model Prediction succeeded for stream and first response: {stream_first_res}")

  def _run_test(self):
    """Test the model locally by making a prediction."""
    # Create the model
    model = self.builder.create_model_instance()
    # send an inference.
    self._run_model_inference(model)

  def test_model(self):
    """Test the model by running it locally in the virtual environment."""

    import_path = repr(os.path.dirname(os.path.abspath(__file__)))
    model_path = repr(self.model_path)

    command_string = (f"import sys; "
                      f"sys.path.append({import_path}); "
                      f"from model_run_locally import ModelRunLocally; "
                      f"ModelRunLocally({model_path})._run_test()")

    command = [self.python_executable, "-c", command_string]
    process = None
    try:
      logger.info("Testing the model locally...")
      process = subprocess.Popen(command)
      # Wait for the process to complete
      process.wait()
      if process.returncode == 0:
        logger.info("Model tested successfully!")
      if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    except subprocess.CalledProcessError as e:
      logger.error(f"Error testing the model: {e}")
      sys.exit(1)
    except Exception as e:
      logger.error(f"Unexpected error: {e}")
      sys.exit(1)
    finally:
      # After the function runs, check if the process is still running
      if process and process.poll() is None:
        logger.info("Process is still running. Terminating process.")
        process.terminate()
        try:
          process.wait(timeout=5)
        except subprocess.TimeoutExpired:
          logger.info("Process did not terminate gracefully. Killing process.")
          # Kill the process if it doesn't terminate after 5 seconds
          process.kill()

  # run the model server
  def run_model_server(self, port=8080):
    """Run the Clarifai Runners's model server."""

    command = [
        self.python_executable, "-m", "clarifai.runners.server", "--model_path", self.model_path,
        "--grpc", "--port",
        str(port)
    ]
    try:
      logger.info(
          f"Starting model server at localhost:{port} with the model at {self.model_path}...")
      subprocess.check_call(command)
      logger.info("Model server started successfully and running at localhost:{port}")
    except subprocess.CalledProcessError as e:
      logger.error(f"Error running model server: {e}")
      self.clean_up()
      sys.exit(1)

  def _docker_hash(self):
    """Generate a hash of the combined requirements file and Dockefile"""
    with open(self.requirements_file, "r") as f:
      requirements_hash = hashlib.md5(f.read().encode('utf-8')).hexdigest()
    with open(os.path.join(self.model_path, "Dockerfile"), "r") as f:
      dockerfile_hash = hashlib.md5(f.read().encode('utf-8')).hexdigest()

    return hashlib.md5(f"{requirements_hash}{dockerfile_hash}".encode('utf-8')).hexdigest()

  def is_docker_installed(self):
    """Checks if Docker is installed on the system."""
    try:
      logger.info("Checking if Docker is installed...")
      subprocess.run(["docker", "--version"], check=True)
      return True
    except subprocess.CalledProcessError:
      logger.error(
          "Docker is not installed! Please install Docker to run the model in a container.")
      return False

  def build_docker_image(
      self,
      image_name="model_image",
  ):
    """Build the docker image using the Dockerfile in the model directory."""
    try:
      logger.info(f"Building docker image from Dockerfile in {self.model_path}...")

      # since we don't want to copy the model directory into the container, we need to modify the Dockerfile and comment out the COPY instruction
      dockerfile_path = os.path.join(self.model_path, "Dockerfile")
      # Read the Dockerfile
      with open(dockerfile_path, 'r') as file:
        lines = file.readlines()

      # Comment out the COPY instruction that copies the current folder
      modified_lines = []
      for line in lines:
        if 'COPY' in line and '/home/nonroot/main' in line:
          modified_lines.append(f'# {line}')
        elif 'download-checkpoints' in line and '/home/nonroot/main' in line:
          modified_lines.append(f'# {line}')
        else:
          modified_lines.append(line)

      # Create a temporary directory to store the modified Dockerfile
      with tempfile.TemporaryDirectory() as temp_dir:
        temp_dockerfile_path = os.path.join(temp_dir, "Dockerfile.temp")

        # Write the modified Dockerfile to the temporary file
        with open(temp_dockerfile_path, 'w') as file:
          file.writelines(modified_lines)

        # Build the Docker image using the temporary Dockerfile
        subprocess.check_call(
            ['docker', 'build', '-t', image_name, '-f', temp_dockerfile_path, self.model_path])
      logger.info(f"Docker image '{image_name}' built successfully!")
    except subprocess.CalledProcessError as e:
      logger.info(f"Error occurred while building the Docker image: {e}")
      sys.exit(1)

  def docker_image_exists(self, image_name):
    """Check if the Docker image exists."""
    try:
      logger.info(f"Checking if Docker image '{image_name}' exists...")
      subprocess.run(["docker", "inspect", image_name], check=True)
      logger.info(f"Docker image '{image_name}' exists!")
      return True
    except subprocess.CalledProcessError:
      logger.info(f"Docker image '{image_name}' does not exist!")
      return False

  def _gpu_is_available(self):
    """
    Checks if nvidia-smi is available, indicating a GPU is likely accessible.
    """
    return shutil.which("nvidia-smi") is not None

  def run_docker_container(self,
                           image_name,
                           container_name="clarifai-model-container",
                           port=8080,
                           env_vars=None):
    """Runs a Docker container from the specified image."""
    try:
      logger.info(f"Running Docker container '{container_name}' from image '{image_name}'...")
      # Base docker run command
      cmd = ["docker", "run", "--name", container_name, '--rm', "--network", "host"]
      if self._gpu_is_available():
        cmd.extend(["--gpus", "all"])
      # Add volume mappings
      cmd.extend(["-v", f"{self.model_path}:/home/nonroot/main"])
      # Add environment variables
      if env_vars:
        for key, value in env_vars.items():
          cmd.extend(["-e", f"{key}={value}"])
      # Add the image name
      cmd.append(image_name)
      # update the CMD to run the server
      cmd.extend(["--model_path", "/home/nonroot/main", "--grpc", "--port", str(port)])
      # Run the container
      process = subprocess.Popen(cmd,)
      logger.info(
          f"Docker container '{container_name}' is running successfully! access the model at http://localhost:{port}"
      )

      # Function to handle Ctrl+C (SIGINT) gracefully
      def signal_handler(sig, frame):
        logger.info(f"Stopping Docker container '{container_name}'...")
        subprocess.run(["docker", "stop", container_name], check=True)
        process.terminate()
        logger.info(f"Docker container '{container_name}' stopped successfully!")
        time.sleep(1)
        sys.exit(0)

      # Register the signal handler for SIGINT (Ctrl+C)
      signal.signal(signal.SIGINT, signal_handler)
      # Wait for the process to finish (keeps the container running until it's stopped)
      process.wait()
    except subprocess.CalledProcessError as e:
      logger.info(f"Error occurred while running the Docker container: {e}")
      sys.exit(1)
    except Exception as e:
      logger.info(f"Error occurred while running the Docker container: {e}")
      sys.exit(1)

  def test_model_container(self,
                           image_name,
                           container_name="clarifai-model-container",
                           env_vars=None):
    """Test the model inside the Docker container."""
    try:
      logger.info("Testing the model inside the Docker container...")
      # Base docker run command
      cmd = ["docker", "run", "--name", container_name, '--rm', "--network", "host"]
      if self._gpu_is_available():
        cmd.extend(["--gpus", "all"])
      # update the entrypoint for testing the model
      cmd.extend(["--entrypoint", "python"])
      # Add volume mappings
      cmd.extend(["-v", f"{self.model_path}:/home/nonroot/main"])
      # Add environment variables
      if env_vars:
        for key, value in env_vars.items():
          cmd.extend(["-e", f"{key}={value}"])
      # Add the image name
      cmd.append(image_name)
      # update the CMD to test the model inside the container
      cmd.extend([
          "-c",
          "from clarifai.runners.models.model_run_locally import ModelRunLocally; ModelRunLocally('/home/nonroot/main')._run_test()"
      ])
      # Run the container
      subprocess.check_call(cmd)
      logger.info("Model tested successfully!")
    except subprocess.CalledProcessError as e:
      logger.error(f"Error testing the model inside the Docker container: {e}")
      sys.exit(1)

  def container_exists(self, container_name="clarifai-model-container"):
    """Check if the Docker container exists."""
    try:
      # Run docker ps -a to list all containers (running and stopped)
      result = subprocess.run(
          ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
          check=True,
          capture_output=True,
          text=True)
      # If the container name is returned, it exists
      if result.stdout.strip() == container_name:
        logger.info(f"Docker container '{container_name}' exists.")
        return True
      else:
        return False
    except subprocess.CalledProcessError as e:
      logger.error(f"Error occurred while checking if container exists: {e}")
      return False

  def stop_docker_container(self, container_name="clarifai-model-container"):
    """Stop the Docker container if it's running."""
    try:
      # Check if the container is running
      result = subprocess.run(
          ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
          check=True,
          capture_output=True,
          text=True)
      if result.stdout.strip() == container_name:
        logger.info(f"Docker container '{container_name}' is running. Stopping it...")
        subprocess.run(["docker", "stop", container_name], check=True)
        logger.info(f"Docker container '{container_name}' stopped successfully!")
    except subprocess.CalledProcessError as e:
      logger.error(f"Error occurred while stopping the Docker container: {e}")

  def remove_docker_container(self, container_name="clarifai-model-container"):
    """Remove the Docker container."""
    try:
      logger.info(f"Removing Docker container '{container_name}'...")
      subprocess.run(["docker", "rm", container_name], check=True)
      logger.info(f"Docker container '{container_name}' removed successfully!")
    except subprocess.CalledProcessError as e:
      logger.error(f"Error occurred while removing the Docker container: {e}")

  def remove_docker_image(self, image_name):
    """Remove the Docker image."""
    try:
      logger.info(f"Removing Docker image '{image_name}'...")
      subprocess.run(["docker", "rmi", image_name], check=True)
      logger.info(f"Docker image '{image_name}' removed successfully!")
    except subprocess.CalledProcessError as e:
      logger.error(f"Error occurred while removing the Docker image: {e}")

  def clean_up(self):
    """Clean up the temporary virtual environment."""
    if os.path.exists(self.temp_dir):
      logger.info("Cleaning up temporary virtual environment...")
      shutil.rmtree(self.temp_dir)


def main(model_path,
         run_model_server=False,
         inside_container=False,
         port=8080,
         keep_env=False,
         keep_image=False,
         skip_dockerfile: bool = False):

  manager = ModelRunLocally(model_path)
  # get whatever stage is in config.yaml to force download now
  # also always write to where upload/build wants to, not the /tmp folder that runtime stage uses
  _, _, _, when, _, _ = manager.builder._validate_config_checkpoints()
  manager.builder.download_checkpoints(
      stage=when, checkpoint_path_override=manager.builder.checkpoint_path)
  if inside_container:
    if not manager.is_docker_installed():
      sys.exit(1)
    if not skip_dockerfile:
      manager.builder.create_dockerfile()
    image_tag = manager._docker_hash()
    model_id = manager.config['model']['id'].lower()
    # must be in lowercase
    image_name = f"{model_id}:{image_tag}"
    container_name = model_id
    if not manager.docker_image_exists(image_name):
      manager.build_docker_image(image_name=image_name)
    try:
      if run_model_server:
        manager.run_docker_container(
            image_name=image_name, container_name=container_name, port=port)
      else:
        manager.test_model_container(image_name=image_name, container_name=container_name)
    finally:
      if manager.container_exists(container_name):
        manager.stop_docker_container(container_name)
        manager.remove_docker_container(container_name=container_name)
      if not keep_image:
        manager.remove_docker_image(image_name)

  else:
    try:
      use_existing_env = manager.create_temp_venv()
      if not use_existing_env:
        manager.install_requirements()
      if run_model_server:
        manager.run_model_server(port)
      else:
        manager.test_model()
    finally:
      if not keep_env:
        manager.clean_up()
