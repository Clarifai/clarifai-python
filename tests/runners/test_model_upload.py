import os
import shutil
from pathlib import Path

import pytest
import yaml
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client import User
from clarifai.runners.models.model_upload import ModelUploader

MODEL_PATH = os.path.join(os.path.dirname(__file__), "dummy_runner_models")
CLARIFAI_USER_ID = os.environ["CLARIFAI_USER_ID"]
CLARIFAI_PAT = os.environ["CLARIFAI_PAT"]
CREATE_APP_ID = "pytest-model-upload-test"


def check_app_exists():
  """
  Check if the app exists on the user account.
  """
  user = User(
      user_id=CLARIFAI_USER_ID,
      base_url=os.environ.get('CLARIFAI_API_BASE', 'https://api.clarifai.com'),
      pat=CLARIFAI_PAT,
  )
  apps = user.list_apps()
  for app in apps:
    if app.id == CREATE_APP_ID:
      return True
  return False


def create_app():
  """
  Creates a Clarifai app for testing purposes.
  """

  user = User(
      user_id=CLARIFAI_USER_ID,
      base_url=os.environ.get('CLARIFAI_API_BASE', 'https://api.clarifai.com'),
      pat=CLARIFAI_PAT,
  )
  if check_app_exists():
    print(f"App '{CREATE_APP_ID}' already exists.")
  else:
    print(f"Creating app '{CREATE_APP_ID}'...")
    user.create_app(app_id=CREATE_APP_ID)
  return CREATE_APP_ID, user


@pytest.fixture(scope="session")
def clarifai_app():
  """
  Fixture to create and clean up a Clarifai app before/after running the tests.
  """
  app_id, user = create_app()
  yield app_id  # Provide the app_id to the tests
  # Cleanup: delete the app after tests
  try:
    user.delete_app(app_id=app_id)
    print(f"Deleted app '{app_id}' successfully.")
  except Exception as e:
    print(f"Failed to delete app '{app_id}': {e}")


@pytest.fixture
def dummy_models_path(tmp_path, clarifai_app):
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
  config["model"]["app_id"] = clarifai_app

  # Rewrite config.yaml
  with config_yaml_path.open("w") as f:
    yaml.dump(config, f, sort_keys=False)

  return str(target_folder)


@pytest.fixture
def model_uploader(dummy_models_path):
  """
  Returns a ModelUploader instance for general usage in tests.
  """
  uploader = ModelUploader(folder=dummy_models_path, validate_api_ids=False)
  return uploader


def test_init_valid_folder(model_uploader):
  """
  Ensure that creating a ModelUploader with a valid folder
  does not raise any exceptions and sets up the object correctly.
  """
  assert os.path.exists(model_uploader.folder)
  assert "config.yaml" in os.listdir(model_uploader.folder)


def test_model_uploader_flow(dummy_models_path):
  """
  End-to-end test that:
  1. Initializes the ModelUploader on the dummy_runner_models folder
  2. Checks folder validation
  3. Creates or reuses an existing model
  4. Uploads a new model version
  5. Waits for the build
  """
  # Initialize
  uploader = ModelUploader(folder=str(dummy_models_path))
  assert uploader.folder == str(dummy_models_path), "Uploader folder mismatch"

  # Basic checks on config
  assert uploader.config["model"]["id"] == "dummy-runner-model"
  assert uploader.config["model"]["user_id"] == os.environ["CLARIFAI_USER_ID"]
  # The app_id should be updated to the newly created ephemeral one
  assert uploader.config["model"]["app_id"] == CREATE_APP_ID

  # Validate that the model doesn't exist yet
  # Because we are using a new ephemeral app, it's unlikely to exist
  assert uploader.check_model_exists() is False, "Model should not exist on new ephemeral app"

  # Create the model (on Clarifai side)
  create_resp = uploader.maybe_create_model()

  if create_resp:
    returned_code = create_resp.status.code
    assert returned_code in [
        status_code_pb2.SUCCESS,
    ], f"Model creation failed with {returned_code}"

  # Now the model should exist
  assert uploader.check_model_exists() is True, "Model should exist after creation"

  # Create the Dockerfile (not crucial for the actual build, but tested in the script)
  uploader.create_dockerfile()
  dockerfile_path = Path(uploader.folder) / "Dockerfile"
  assert dockerfile_path.exists(), "Dockerfile was not created."

  # Upload a new version
  uploader.upload_model_version(download_checkpoints=False)

  # After starting the upload/build, we expect model_version_id to be set if it began building
  assert uploader.model_version_id is not None, "Model version upload failed to initialize"

  print(f"Test completed successfully with model_version_id={uploader.model_version_id}")
