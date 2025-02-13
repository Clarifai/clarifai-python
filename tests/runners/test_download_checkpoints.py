import os
import shutil
import tempfile

import pytest

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.utils.const import DEFAULT_RUNTIME_DOWNLOAD_PATH
from clarifai.runners.utils.loader import HuggingFaceLoader

MODEL_ID = "timm/mobilenetv3_small_100.lamb_in1k"


@pytest.fixture(scope="module")
def checkpoint_dir():
  # Create a temporary directory for the test checkpoints
  temp_dir = os.path.join(tempfile.gettempdir(), MODEL_ID[5:])
  if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
  yield temp_dir  # Provide the directory to the tests
  # Cleanup: remove the directory after all tests are complete
  shutil.rmtree(temp_dir, ignore_errors=True)


# Pytest fixture to delete the checkpoints in dummy runner models folder after tests complete
@pytest.fixture(scope="function")
def dummy_runner_models_dir():
  model_folder_path = os.path.join(os.path.dirname(__file__), "dummy_runner_models")
  checkpoints_path = os.path.join(model_folder_path, "1", "checkpoints")
  yield checkpoints_path
  # Cleanup the checkpoints folder after the test
  if os.path.exists(checkpoints_path):
    shutil.rmtree(checkpoints_path)


@pytest.fixture(scope="function", autouse=True)
def override_environment_variables():
  # Backup the existing environment variable value
  original_clarifai_pat = os.environ.get("CLARIFAI_PAT")
  if "CLARIFAI_PAT" in os.environ:
    del os.environ["CLARIFAI_PAT"]  # Temporarily unset the variable for the tests
  yield
  # Restore the original environment variable value after tests
  if original_clarifai_pat:
    os.environ["CLARIFAI_PAT"] = original_clarifai_pat


def test_loader_download_checkpoints(checkpoint_dir):
  loader = HuggingFaceLoader(repo_id=MODEL_ID)
  loader.download_checkpoints(checkpoint_path=checkpoint_dir)
  assert len(os.listdir(checkpoint_dir)) == 4


def test_validate_download(checkpoint_dir):
  loader = HuggingFaceLoader(repo_id=MODEL_ID)
  assert loader.validate_download(checkpoint_path=checkpoint_dir) is True


def test_download_checkpoints(dummy_runner_models_dir):
  # This doesn't have when in it's config.yaml so runtime.
  model_folder_path = os.path.join(os.path.dirname(__file__), "dummy_runner_models")
  model_builder = ModelBuilder(model_folder_path, download_validation_only=True)
  # defaults to runtime stage which matches config.yaml not having a when field.
  # get whatever stage is in config.yaml to force download now
  # also always write to where upload/build wants to, not the /tmp folder that runtime stage uses
  _, _, _, when = model_builder._validate_config_checkpoints()
  checkpoint_dir = model_builder.download_checkpoints(
      stage=when, checkpoint_path=model_builder.checkpoint_path)
  assert checkpoint_dir == DEFAULT_RUNTIME_DOWNLOAD_PATH

  # This doesn't have when in it's config.yaml so build.
  model_folder_path = os.path.join(os.path.dirname(__file__), "hf_mbart_model")
  model_builder = ModelBuilder(model_folder_path, download_validation_only=True)
  # defaults to runtime stage which matches config.yaml not having a when field.
  # get whatever stage is in config.yaml to force download now
  # also always write to where upload/build wants to, not the /tmp folder that runtime stage uses
  _, _, _, when = model_builder._validate_config_checkpoints()
  checkpoint_dir = model_builder.download_checkpoints(
      stage=when, checkpoint_path=model_builder.checkpoint_path)
  assert checkpoint_dir == os.path.join(
      os.path.dirname(__file__), "hf_mbart_model", "1", "checkpoints")
