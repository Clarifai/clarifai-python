import os
import pytest
from clarifai_grpc.grpc.api import resources_pb2

from clarifai.client.workflow import Workflow

DOG_IMAGE_URL = "https://samples.clarifai.com/dog2.jpeg"
NON_EXISTING_IMAGE_URL = "http://example.com/non-existing.jpg"
RED_TRUCK_IMAGE_FILE_PATH = "tests/assets/red-truck.png"
BEER_VIDEO_URL = "https://samples.clarifai.com/beer.mp4"

MAIN_APP_ID = "main"
MAIN_APP_USER_ID = "clarifai"
WORKFLOW_ID = "General"

CLARIFAI_PAT = os.environ["CLARIFAI_PAT"]


@pytest.fixture
def workflow():
  return Workflow(
      user_id=MAIN_APP_USER_ID,
      app_id=MAIN_APP_ID,
      workflow_id=WORKFLOW_ID,
      output_config=resources_pb2.OutputConfig(max_concepts=3),
      pat=CLARIFAI_PAT)


@pytest.mark.requires_secrets
class TestWorkflowPredict:

  def test_workflow_predict_image_url(self, workflow):
    post_workflows_response = workflow.predict_by_url(DOG_IMAGE_URL, input_type="image")

    assert len(post_workflows_response.results[0].outputs[0].data.concepts) > 0

  def test_workflow_predict_image_bytes(self, workflow):
    with open(RED_TRUCK_IMAGE_FILE_PATH, "rb") as f:
      file_bytes = f.read()
    post_workflows_response = workflow.predict_by_bytes(file_bytes, input_type="image")

    assert len(post_workflows_response.results[0].outputs[0].data.concepts) > 0

  def test_workflow_predict_max_concepts(self):
    workflow = Workflow(
        user_id=MAIN_APP_USER_ID,
        app_id=MAIN_APP_ID,
        workflow_id=WORKFLOW_ID,
        output_config=resources_pb2.OutputConfig(max_concepts=3))
    post_workflows_response = workflow.predict_by_url(DOG_IMAGE_URL, input_type="image")

    assert len(post_workflows_response.results[0].outputs[0].data.concepts) == 3
