import logging
import os

import pytest

from clarifai.client.app import App
from clarifai.client.user import User

MAIN_APP_ID = "main"
MAIN_APP_USER_ID = "clarifai"
GENERAL_MODEL_ID = "general-image-recognition"
General_Workflow_ID = "General"

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]
CREATE_APP_ID = "ci_test_app"
CREATE_MODEL_ID = "ci_test_model"
CREATE_WORKFLOW_ID = "ci_test_workflow"
CREATE_DATASET_ID = "ci_test_dataset"
CREATE_MODULE_ID = "ci_test_module"


@pytest.fixture
def create_app():
  return App(user_id=CREATE_APP_USER_ID, app_id=CREATE_APP_ID)


@pytest.fixture
def app():
  return App(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID)


@pytest.fixture
def client():
  return User(user_id=MAIN_APP_USER_ID)


class TestApp:
  """Tests for the App class and its methods.

    CRUD operations are tested for each of the following resources:
    - app
    - dataset
    - model
    - workflow

    Note: Update to be added later.
    """

  def test_list_models(self, app):
    all_models = app.list_models()
    assert len(all_models) > 0

  def test_list_workflows(self, app):
    all_workflows = app.list_workflows()
    assert len(all_workflows) > 0

  def test_list_apps(self, client):
    all_apps = client.list_apps()
    assert len(all_apps) > 0

  def test_get_model(self, client):
    model = client.app(app_id=MAIN_APP_ID).model(model_id=GENERAL_MODEL_ID)
    assert model.id == GENERAL_MODEL_ID and model.app_id == MAIN_APP_ID and model.user_id == MAIN_APP_USER_ID

  def test_get_workflow(self, client):
    workflow = client.app(app_id=MAIN_APP_ID).workflow(workflow_id=General_Workflow_ID)
    assert workflow.id == General_Workflow_ID and workflow.app_id == MAIN_APP_ID and workflow.user_id == MAIN_APP_USER_ID

  def test_create_app(self):
    app = User(user_id=CREATE_APP_USER_ID).create_app(app_id=CREATE_APP_ID)
    assert app.id == CREATE_APP_ID and app.user_id == CREATE_APP_USER_ID

  def test_create_dataset(self, create_app):
    dataset = create_app.create_dataset(CREATE_DATASET_ID)
    assert dataset.id == CREATE_DATASET_ID and dataset.app_id == CREATE_APP_ID and dataset.user_id == CREATE_APP_USER_ID

  def test_create_model(self, create_app):
    model = create_app.create_model(CREATE_MODEL_ID)
    assert model.id == CREATE_MODEL_ID and model.app_id == CREATE_APP_ID and model.user_id == CREATE_APP_USER_ID

  def test_create_workflow(self, create_app):
    workflow = create_app.create_workflow(CREATE_WORKFLOW_ID)
    assert workflow.id == CREATE_WORKFLOW_ID and workflow.app_id == CREATE_APP_ID and workflow.user_id == CREATE_APP_USER_ID

  def test_create_module(self, create_app):
    module = create_app.create_module(CREATE_MODULE_ID, description="CI test module")
    assert module.id == CREATE_MODULE_ID and module.app_id == CREATE_APP_ID and module.user_id == CREATE_APP_USER_ID

  def test_delete_dataset(self, create_app, caplog):
    with caplog.at_level(logging.INFO):
      create_app.delete_dataset(CREATE_DATASET_ID)
      assert "SUCCESS" in caplog.text

  def test_delete_workflow(self, create_app, caplog):
    with caplog.at_level(logging.INFO):
      create_app.delete_workflow(CREATE_WORKFLOW_ID)
      assert "SUCCESS" in caplog.text

  def test_delete_model(self, create_app, caplog):
    with caplog.at_level(logging.INFO):
      create_app.delete_model(CREATE_MODEL_ID)
      assert "SUCCESS" in caplog.text

  def test_delete_module(self, create_app, caplog):
    with caplog.at_level(logging.INFO):
      create_app.delete_module(CREATE_MODULE_ID)
      assert "SUCCESS" in caplog.text

  def test_delete_app(self, caplog):
    with caplog.at_level(logging.INFO):
      User(user_id=CREATE_APP_USER_ID).delete_app(CREATE_APP_ID)
      assert "SUCCESS" in caplog.text

  def test_get_dataset(self):
    pass  # TODO

  def test_list_datasets(self):
    pass  # TODO
