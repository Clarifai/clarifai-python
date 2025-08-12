import os
import shutil
import uuid
from pathlib import Path

import pytest
import yaml
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client import User
from clarifai.runners.models.model_builder import ModelBuilder

MODEL_PATH = os.path.join(os.path.dirname(__file__), "dummy_runner_models")
CLARIFAI_USER_ID = os.environ["CLARIFAI_USER_ID"]
CLARIFAI_PAT = os.environ["CLARIFAI_PAT"]
NOW = uuid.uuid4().hex[:10]
CREATE_APP_ID = f"pytest-model-upload-test-{NOW}"


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


@pytest.fixture(scope="module")
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
        raise FileNotFoundError(
            f"Could not find dummy_runner_models at {original_dummy_path}. "
            "Adjust path or ensure it exists."
        )

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
def model_builder(dummy_models_path):
    """
    Returns a ModelBuilder instance for general usage in tests.
    """
    return ModelBuilder(folder=dummy_models_path, validate_api_ids=False)


def test_init_valid_folder(model_builder):
    """
    Ensure that creating a ModelBuilder with a valid folder
    does not raise any exceptions and sets up the object correctly.
    """
    assert os.path.exists(model_builder.folder)
    assert "config.yaml" in os.listdir(model_builder.folder)


def test_model_uploader_flow(dummy_models_path):
    """
    End-to-end test that:
    1. Initializes the ModelBuilder on the dummy_runner_models folder
    2. Checks folder validation
    3. Creates or reuses an existing model
    4. Uploads a new model version
    5. Waits for the build
    """
    # Initialize
    builder = ModelBuilder(folder=str(dummy_models_path))
    assert builder.folder == str(dummy_models_path), "Uploader folder mismatch"

    # Basic checks on config
    assert builder.config["model"]["id"] == "dummy-runner-model"
    assert builder.config["model"]["user_id"] == os.environ["CLARIFAI_USER_ID"]
    # The app_id should be updated to the newly created ephemeral one
    assert builder.config["model"]["app_id"] == CREATE_APP_ID

    # # Validate that the model doesn't exist yet
    # # Because we are using a new ephemeral app, it's unlikely to exist
    # assert builder.check_model_exists() is False, "Model should not exist on new ephemeral app"

    # Create the model (on Clarifai side)
    create_resp = builder.maybe_create_model()

    if create_resp:
        returned_code = create_resp.status.code
        assert returned_code in [
            status_code_pb2.SUCCESS,
        ], f"Model creation failed with {returned_code}"

    # Now the model should exist
    assert builder.check_model_exists() is True, "Model should exist after creation"

    # Create the Dockerfile (not crucial for the actual build, but tested in the script)
    builder.create_dockerfile(generate_dockerfile=True)
    dockerfile_path = Path(builder.folder) / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile was not created."

    # Upload a new version
    builder.upload_model_version()

    # After starting the upload/build, we expect model_version_id to be set if it began building
    assert builder.model_version_id is not None, "Model version upload failed to initialize"

    print(f"Test completed successfully with model_version_id={builder.model_version_id}")


@pytest.fixture
def my_tmp_path(tmp_path):
    return tmp_path


def test_model_uploader_missing_app_action(tmp_path, monkeypatch):
    tests_dir = Path(__file__).parent.resolve()
    original_dummy_path = tests_dir / "dummy_runner_models"

    if not original_dummy_path.exists():
        # Adjust or raise an error if you cannot locate the dummy_runner_models folder
        raise FileNotFoundError(
            f"Could not find dummy_runner_models at {original_dummy_path}. "
            "Adjust path or ensure it exists."
        )

    # Copy the entire folder to tmp_path
    target_folder = tmp_path / "dummy_runner_models"
    shutil.copytree(original_dummy_path, target_folder)

    config_yaml_path = target_folder / "config.yaml"
    with config_yaml_path.open("r") as f:
        config = yaml.safe_load(f)

    # Update the config.yaml to override the app_id with the ephemeral one
    NOW = uuid.uuid4().hex[:8]
    user_id = CLARIFAI_USER_ID
    config["model"]["user_id"] = user_id
    user = User(user_id=user_id)
    # Change app id to non existing one.

    new_app_id = "ci_test_model_upload" + NOW
    config["model"]["app_id"] = new_app_id

    # Rewrite config.yaml
    with config_yaml_path.open("w") as f:
        yaml.dump(config, f, sort_keys=False)

    # ensure app not existing
    with pytest.raises(Exception):
        user.app(app_id=new_app_id)

    # ----- Start testing ----- #

    # With prompt
    # Not create app
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(SystemExit) as exc_info:
        ModelBuilder(target_folder, app_not_found_action="prompt")

    # Create app
    monkeypatch.setattr("builtins.input", lambda _: "y")
    ModelBuilder(target_folder, app_not_found_action="prompt")
    # app must exist
    user.app(app_id=new_app_id)
    user.delete_app(app_id=new_app_id)

    # Without prompt

    # Not go through as app not existing
    with pytest.raises(SystemExit) as exc_info:
        ModelBuilder(target_folder, app_not_found_action="error")
    with pytest.raises(Exception):
        user.app(app_id=new_app_id)

    # Go through
    ModelBuilder(target_folder, app_not_found_action="auto_create")
    user.app(app_id=new_app_id)
    user.delete_app(app_id=new_app_id)

    # Test non supported action
    with pytest.raises(AssertionError):
        ModelBuilder(target_folder, app_not_found_action="a")


def test_openai_stream_options_validation(tmp_path):
    """
    Test that OpenAI models without proper stream_options configuration are rejected.
    """
    tests_dir = Path(__file__).parent.resolve()
    original_dummy_path = tests_dir / "dummy_missing_stream_options_model"

    if not original_dummy_path.exists():
        pytest.skip(
            "dummy_missing_stream_options_model not found, skipping OpenAI validation test"
        )

    # Copy the OpenAI model folder to tmp_path
    target_folder = tmp_path / "dummy_missing_stream_options_model"
    shutil.copytree(original_dummy_path, target_folder)

    # Update config.yaml with test user/app info
    config_yaml_path = target_folder / "config.yaml"
    with config_yaml_path.open("r") as f:
        config = yaml.safe_load(f)

    NOW = uuid.uuid4().hex[:8]
    config["model"]["user_id"] = CLARIFAI_USER_ID
    config["model"]["app_id"] = "test-openai-validation" + NOW

    with config_yaml_path.open("w") as f:
        yaml.dump(config, f, sort_keys=False)

    # Test that ModelBuilder raises exception for missing stream_options
    with pytest.raises(Exception) as exc_info:
        ModelBuilder(str(target_folder), validate_api_ids=False)

    # Verify the exception message contains the expected validation error
    assert "include_usage" in str(exc_info.value)
    assert "stream_options" in str(exc_info.value)
