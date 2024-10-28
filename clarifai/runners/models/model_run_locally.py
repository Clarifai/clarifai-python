import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import venv

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from clarifai_protocol import BaseRunner

from clarifai.runners.models.model_upload import ModelUploader
from clarifai.utils.logging import logger


class ModelRunLocally:

  def __init__(self, model_path):
    self.model_path = model_path
    self.requirements_file = os.path.join(self.model_path, "requirements.txt")

  def create_temp_venv(self):
    """Create a temporary virtual environment."""
    logger.info("Creating temporary virtual environment...")
    temp_dir = tempfile.mkdtemp()
    venv_dir = os.path.join(temp_dir, "venv")
    venv.create(venv_dir, with_pip=True)

    self.venv_dir = venv_dir
    self.temp_dir = temp_dir
    self.python_executable = os.path.join(venv_dir, "bin", "python")

    logger.info(f"Created temporary virtual environment at {venv_dir}")
    return venv_dir, temp_dir

  def install_requirements(self):
    """Install the dependencies from requirements.txt and Clarifai."""
    pip_executable = os.path.join(self.venv_dir, "bin", "pip")
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

  def _get_model_runner(self):
    """Dynamically import the runner class from the model file."""

    # import the runner class that to be implement by the user
    runner_path = os.path.join(self.model_path, "1", "model.py")

    # arbitrary name given to the module to be imported
    module = "runner_module"

    spec = importlib.util.spec_from_file_location(module, runner_path)
    runner_module = importlib.util.module_from_spec(spec)
    sys.modules[module] = runner_module
    spec.loader.exec_module(runner_module)

    # Find all classes in the model.py file that are subclasses of BaseRunner
    classes = [
        cls for _, cls in inspect.getmembers(runner_module, inspect.isclass)
        if issubclass(cls, BaseRunner) and cls.__module__ == runner_module.__name__
    ]

    #  Ensure there is exactly one subclass of BaseRunner in the model.py file
    if len(classes) != 1:
      raise Exception("Expected exactly one subclass of BaseRunner, found: {}".format(
          len(classes)))

    MyRunner = classes[0]
    return MyRunner

  def _build_request(self):
    """Create a mock inference request for testing the model."""

    uploader = ModelUploader(self.model_path)
    model_version_proto = uploader.get_model_version_proto()
    model_version_proto.id = "model_version"

    return service_pb2.PostModelOutputsRequest(
        model=resources_pb2.Model(model_version=model_version_proto),
        inputs=[
            resources_pb2.Input(data=resources_pb2.Data(
                text=resources_pb2.Text(raw="How many people live in new york?"),
                image=resources_pb2.Image(url="https://samples.clarifai.com/metro-north.jpg"),
                audio=resources_pb2.Audio(url="https://samples.clarifai.com/GoodMorning.wav"),
            ))
        ],
    )

  def _run_model_inference(self, runner):
    """Perform inference using the runner."""
    request = self._build_request()

    try:
      return runner.predict(request)
    except Exception as e:
      logger.error(f"Model Prediction failed: {e}")
      traceback.print_exc()
      return service_pb2.MultiOutputResponse(status=status_pb2.Status(
          code=status_code_pb2.MODEL_PREDICTION_FAILED,
          description="Prediction failed",
          details="",
          internal_details=str(e),
      ))

  def _run_test(self):
    """Test the model locally by making a prediction."""
    # validate that we have checkpoints downloaded before constructing MyRunner
    uploader = ModelUploader(self.model_path)
    uploader.download_checkpoints()
    # construct MyRunner which will call load_model()
    MyRunner = self._get_model_runner()
    runner = MyRunner(
        runner_id="n/a",
        nodepool_id="n/a",
        compute_cluster_id="n/a",
        user_id="n/a",
    )

    # send an inference.
    response = self._run_model_inference(runner)
    if response.outputs[0].status.code != status_code_pb2.SUCCESS:
      logger.error(f"Moddel Prediction failed: {response}")
    else:
      logger.info(f"Model Prediction succeeded: {response}")

  def test_model(self):
    """Test the model by running it locally in the virtual environment."""
    command = [
        self.python_executable,
        "-c",
        f"import sys; sys.path.append('{os.path.dirname(os.path.abspath(__file__))}'); "
        f"from model_run_locally import ModelRunLocally; ModelRunLocally('{self.model_path}')._run_test()",
    ]
    try:
      logger.info("Testing the model locally...")
      subprocess.check_call(command)
      logger.info("Model tested successfully!")
    except subprocess.CalledProcessError as e:
      logger.error(f"Error testing the model: {e}")
      sys.exit(1)

  # run the model server
  def run_model_server(self):
    """Run the Clarifai Runners's model server."""

    command = [
        self.python_executable, "-m", "clarifai.runners.server", "--model_path", self.model_path,
        "--start_dev_server"
    ]
    try:
      logger.info(f"Starting model server with model at {self.model_path}...")
      subprocess.check_call(command)
      logger.info("Model server started successfully!")
    except subprocess.CalledProcessError as e:
      logger.error(f"Error running model server: {e}")
      self.clean_up()
      sys.exit(1)

  def clean_up(self):
    """Clean up the temporary virtual environment."""
    if os.path.exists(self.temp_dir):
      logger.info("Cleaning up temporary virtual environment...")
      shutil.rmtree(self.temp_dir)


def main(model_path, run_model_server=False):

  manager = ModelRunLocally(model_path)
  manager.create_temp_venv()

  try:
    manager.install_requirements()
    if run_model_server:
      manager.run_model_server()
    else:
      manager.test_model()
  finally:
    manager.clean_up()
