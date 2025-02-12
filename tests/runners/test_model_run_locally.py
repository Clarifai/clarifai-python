import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.models.model_run_locally import ModelRunLocally

CLARIFAI_USER_ID = os.environ["CLARIFAI_USER_ID"]
CLARIFAI_PAT = os.environ["CLARIFAI_PAT"]


@pytest.fixture
def dummy_models_path(tmp_path):
  """
  Copy the dummy_runner_models folder to a temp directory and update app_id in config.yaml
  so that your e2e tests use a newly created ephemeral app on your Clarifai account.
  """
  tests_dir = Path(__file__).parent.resolve()
  original_dummy_path = tests_dir / "dummy_runner_models"
  if not original_dummy_path.exists():
    # Adjust or raise an error if you cannot locate the dummy_runner_models folder
    raise FileNotFoundError(f"Could not find dummy_runner_models at {original_dummy_path}. "
                            "Adjust path or ensure it exists.")

  # Copy the entire folder to tmp_path
  target_folder = tmp_path / "dummy_runner_models"
  shutil.copytree(original_dummy_path, target_folder)

  # Update the config.yaml to override the app_id with the ephemeral one
  config_yaml_path = target_folder / "config.yaml"
  with config_yaml_path.open("r") as f:
    config = yaml.safe_load(f)

  # Overwrite the app_id with the newly created clarifai_app
  config["model"]["user_id"] = CLARIFAI_USER_ID

  # Rewrite config.yaml
  with config_yaml_path.open("w") as f:
    yaml.dump(config, f, sort_keys=False)

  return str(target_folder)


@pytest.fixture
def model_run_locally(dummy_models_path):
  """
  Fixture that instantiates the ModelRunLocally class
  with the dummy model_path that already exists.
  """
  return ModelRunLocally(dummy_models_path)


@pytest.fixture
def dummy_hf_models_path(tmp_path):
  """
  Copy the hf_mbart_model folder to a temp directory and update app_id in config.yaml
  so that your e2e tests use a newly created ephemeral app on your Clarifai account.
  """
  tests_dir = Path(__file__).parent.resolve()
  original_dummy_path = tests_dir / "hf_mbart_model"
  if not original_dummy_path.exists():
    # Adjust or raise an error if you cannot locate the hf_mbart_model folder
    raise FileNotFoundError(f"Could not find hf_mbart_model at {original_dummy_path}. "
                            "Adjust path or ensure it exists.")

  # Copy the entire folder to tmp_path
  target_folder = tmp_path / "hf_mbart_model"
  shutil.copytree(original_dummy_path, target_folder)

  # Update the config.yaml to override the app_id with the ephemeral one
  config_yaml_path = target_folder / "config.yaml"
  with config_yaml_path.open("r") as f:
    config = yaml.safe_load(f)

  # Overwrite the app_id with the newly created clarifai_app
  config["model"]["user_id"] = CLARIFAI_USER_ID

  # Rewrite config.yaml
  with config_yaml_path.open("w") as f:
    yaml.dump(config, f, sort_keys=False)

  return str(target_folder)


@pytest.fixture
def hf_model_run_locally(dummy_hf_models_path):
  """
  Fixture that instantiates the ModelRunLocally class
  with the dummy model_path that already exists.
  """
  return ModelRunLocally(dummy_hf_models_path)


def test_create_model_instance(model_run_locally):
  """
  Test that create_model_instance returns a valid model instance.
  """
  model = model_run_locally.builder.create_model_instance()
  assert model is not None, "Expected a model class to be returned."
  assert isinstance(model, ModelClass), "Expected a model class to be returned."


def test_build_request(model_run_locally):
  """
  Test that _build_request returns a well-formed PostModelOutputsRequest
  """
  request = model_run_locally._build_request()
  assert request is not None
  assert len(request.inputs) == 1, "Expected exactly one input in constructed request."


def test_create_temp_venv(model_run_locally):
  """
  Test whether create_temp_venv correctly initializes a virtual environment directory.
  """
  use_existing_venv = model_run_locally.create_temp_venv()
  # Confirm we get back a boolean
  assert isinstance(use_existing_venv, bool)
  # Check that the venv_dir was set
  venv_dir = Path(model_run_locally.venv_dir)
  assert venv_dir.exists()
  # Clean up
  model_run_locally.clean_up()
  assert not venv_dir.exists(), "Temporary virtual environment was not cleaned up"


def test_install_requirements(model_run_locally):
  """
  Test installing requirements into the virtual environment.
  Note: This actually installs from the dummy requirements.txt.
  """
  # Create the environment
  model_run_locally.create_temp_venv()
  # Attempt to install requirements
  try:
    model_run_locally.install_requirements()
  except SystemExit:
    pytest.fail("install_requirements() failed and exited.")
  # Clean up
  model_run_locally.clean_up()


def test_test_model_success(model_run_locally):
  """
  Test that test_model succeeds with the dummy model.
  This calls the script's test_model method, which runs a subprocess.
  """
  model_run_locally.create_temp_venv()
  model_run_locally.install_requirements()

  # Catch the subprocess call. If the dummy model is correct, exit code should be 0.
  try:
    model_run_locally.test_model()
  except SystemExit:
    pytest.fail("test_model() triggered a system exit with non-zero code.")
  except subprocess.CalledProcessError:
    # If the process didn't return code 0, fail the test
    pytest.fail("The model test did not complete successfully in the subprocess.")
  finally:
    # Clean up
    model_run_locally.clean_up()


def test_hf_test_model_success(hf_model_run_locally):
  """
  Test that test_model succeeds with the dummy model.
  This calls the script's test_model method, which runs a subprocess.
  """
  hf_model_run_locally.builder.download_checkpoints(stage="build")
  hf_model_run_locally.create_temp_venv()
  hf_model_run_locally.install_requirements()

  # Catch the subprocess call. If the dummy model is correct, exit code should be 0.
  try:
    hf_model_run_locally.test_model()
  except SystemExit:
    pytest.fail("test_model() triggered a system exit with non-zero code.")
  except subprocess.CalledProcessError:
    # If the process didn't return code 0, fail the test
    pytest.fail("The model test did not complete successfully in the subprocess.")
  finally:
    # Clean up
    hf_model_run_locally.clean_up()
