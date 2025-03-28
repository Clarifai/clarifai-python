import os
import uuid
import pytest
from clarifai.client.model import Model
from clarifai.client.input import Inputs
from clarifai.errors import ApiError

# Placeholder for the actual model file content/path
# Creating a valid model package on the fly is complex.
# For now, we'll assume a minimal file might work or fail gracefully.
# A better approach might involve a pre-built dummy model artifact.
DUMMY_MODEL_FILENAME = "dummy_model_file.txt"
DUMMY_MODEL_CONTENT = "This is a dummy model file."

@pytest.fixture(scope="module")
def temp_model_file():
    """Creates a temporary dummy model file."""
    filepath = DUMMY_MODEL_FILENAME
    with open(filepath, "w") as f:
        f.write(DUMMY_MODEL_CONTENT)
    yield filepath
    os.remove(filepath)

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
        pytest.skip("Environment variables CLARIFAI_USER_ID, CLARIFAI_APP_ID, CLARIFAI_PAT must be set")
    return {"user_id": user_id, "app_id": app_id, "pat": pat}

# NOTE: The `create_version_by_file` method is complex and expects a specific
# format (likely Triton). This test uses a placeholder file and might fail
# at the upload stage depending on API validation.
# A more robust test would require a pre-built, valid dummy model artifact.
@pytest.mark.skip(reason="create_version_by_file requires a valid model package, placeholder used.")
def test_upload_model_and_predict(temp_model_file, test_model_id, client_credentials):
    """
    Tests uploading a model version using create_version_by_file and then predicting with it.
    """
    model_client = Model(
        user_id=client_credentials["user_id"],
        app_id=client_credentials["app_id"],
        model_id=test_model_id,
        pat=client_credentials["pat"]
    )

    # Define minimal input/output maps (adjust if needed based on expected model type)
    # These are placeholders and depend heavily on the actual model file structure.
    input_field_maps = {"text": "INPUT__0"}
    output_field_maps = {"text": "OUTPUT__0"}

    try:
        # Attempt to create the model version by uploading the dummy file
        # This is the most likely point of failure without a valid model package.
        created_model = model_client.create_version_by_file(
            file_path=temp_model_file,
            input_field_maps=input_field_maps,
            output_field_maps=output_field_maps,
            description="Test model upload"
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
            pat=client_credentials["pat"]
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
            inputs=[input_proto]
        )

        response = predict_client._grpc_request(predict_client.stub.PostModelOutputs, request)

        assert response.status.code == status_code_pb2.SUCCESS
        assert len(response.outputs) == 1
        # Add more specific assertions based on the expected output of the dummy model
        # For a simple text echo model, output text might match input text.
        # This depends entirely on the (currently non-functional) dummy model.
        # assert response.outputs[0].data.text.raw == input_text

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
