import os
import uuid

import pytest

from clarifai.client.input import Inputs
from clarifai.client.model import Model
from clarifai.errors import ApiError

# Placeholder for the actual model file content/path
# Creating a valid model package on the fly is complex.
# For now, we'll assume a minimal file might work or fail gracefully.
# Use the existing dummy runner model structure.
DUMMY_MODEL_DIR = "tests/runners/dummy_runner_models/1"


@pytest.fixture(scope="module")
def dummy_model_path():
    """Provides the path to the dummy model directory."""
    # Ensure the directory exists before returning the path
    if not os.path.isdir(DUMMY_MODEL_DIR):
        pytest.skip(f"Dummy model directory not found at {DUMMY_MODEL_DIR}")
    return DUMMY_MODEL_DIR


@pytest.fixture(scope="module")
def test_model_id():
    """Generates a unique model ID for the test."""
    return f"test-upload-predict-{uuid.uuid4()}"


# Fixture to get credentials from environment variables
# Assumes CLARIFAI_USER_ID, CLARIFAI_APP_ID, CLARIFAI_PAT are set
@pytest.fixture(scope="module")
def client_credentials():
    user_id = os.getenv("CLARIFAI_USER_ID")
    app_id = os.getenv("CLARIFAI_APP_ID")
    pat = os.getenv("CLARIFAI_PAT")
    if not all([user_id, app_id, pat]):
        pytest.skip(
            "Environment variables CLARIFAI_USER_ID, CLARIFAI_APP_ID, CLARIFAI_PAT must be set"
        )
    return {"user_id": user_id, "app_id": app_id, "pat": pat}


# NOTE: The `create_version_by_file` method is complex and expects a specific
# format (likely Triton). This test uses the dummy_runner_model.
def test_upload_model_and_predict(dummy_model_path, test_model_id, client_credentials):
    """
    Tests uploading a model version using create_version_by_file with a directory
    and then predicting with it. Assumes the dummy model echoes text input.
    """
    model_client = Model(
        user_id=client_credentials["user_id"],
        app_id=client_credentials["app_id"],
        model_id=test_model_id,
        pat=client_credentials["pat"],
    )

    # Define minimal input/output maps (adjust if needed based on expected model type)
    # These are placeholders and depend heavily on the actual model file structure.
    input_field_maps = {"text": "INPUT__0"}
    output_field_maps = {"text": "OUTPUT__0"}

    try:
        # Attempt to create the model version by uploading the dummy model directory.
        # The client should package this directory into a tarball.
        created_model = model_client.create_version_by_file(
            file_path=dummy_model_path,  # Use the directory path
            input_field_maps=input_field_maps,
            output_field_maps=output_field_maps,
            description="Test model upload",
        )

        assert created_model is not None
        assert created_model.model_id == test_model_id
        assert created_model.model_version is not None
        assert created_model.model_version.get("id") is not None

        # If upload succeeds, attempt prediction
        # Use the specific version ID from the upload response
        predict_client = Model(
            user_id=client_credentials["user_id"],
            app_id=client_credentials["app_id"],
            model_id=test_model_id,
            model_version={'id': created_model.model_version["id"]},
            pat=client_credentials["pat"],
        )

        # Prepare input data (e.g., text)
        input_text = "Test prediction input."
        input_proto = Inputs.get_text_input(input_id=str(uuid.uuid4()), raw_text=input_text)

        # Make prediction request
        # Note: predict_by_bytes/url might be simpler if the model supports standard inputs
        # Using the lower-level _grpc_request with PostModelOutputsRequest for flexibility
        from clarifai_grpc.grpc.api import service_pb2
        from clarifai_grpc.grpc.api.status import status_code_pb2

        request = service_pb2.PostModelOutputsRequest(
            user_app_id=predict_client.auth_helper.get_user_app_id_proto(),
            model_id=predict_client.model_id,
            version_id=predict_client.model_version["id"],
            inputs=[input_proto],
        )

        response = predict_client._grpc_request(predict_client.stub.PostModelOutputs, request)

        assert response.status.code == status_code_pb2.SUCCESS
        assert len(response.outputs) == 1
        # Assert based on the expected behavior of the dummy echo model.
        assert response.outputs[0].data.text.raw == input_text

    except ApiError as e:
        # Catch potential API errors during upload or prediction
        pytest.fail(f"API Error occurred: {e}")

    except Exception as e:
        # Catch any other unexpected errors
        pytest.fail(f"An unexpected error occurred: {e}")

    finally:
        # Optional: Clean up the created model version
        # This requires delete permissions and might be complex if versions are immutable
        # Consider leaving test models in a dedicated test app instead.
        pass
