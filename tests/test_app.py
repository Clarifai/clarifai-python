import logging
import os
import uuid

import pytest

from clarifai.client.app import App
from clarifai.client.user import User
from clarifai.constants.search import DEFAULT_TOP_K

NOW = uuid.uuid4().hex[:10]
MAIN_APP_ID = "main"
MAIN_APP_USER_ID = "clarifai"
GENERAL_MODEL_ID = "general-image-recognition"
General_Workflow_ID = "General"

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]
CREATE_APP_ID = f"ci_test_app_{NOW}"
CREATE_MODEL_ID = f"ci_test_model_{NOW}"
CREATE_DATASET_ID = f"ci_test_dataset_{NOW}"
CREATE_MODULE_ID = f"ci_test_module_{NOW}"
CREATE_RUNNER_ID = f"ci_test_runner_{NOW}"

#assets
IMAGE_URL = "https://samples.clarifai.com/metro-north.jpg"
SUBJECT_CONCEPT_ID = 'honey'
OBJECT_CONCEPT_ID = 'food'
PREDICATE = "hypernym"

CLARIFAI_PAT = os.environ["CLARIFAI_PAT"]


@pytest.fixture
def create_app():
  return App(user_id=CREATE_APP_USER_ID, app_id=CREATE_APP_ID, pat=CLARIFAI_PAT)


@pytest.fixture
def app():
  return App(user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID, pat=CLARIFAI_PAT)


@pytest.fixture
def client():
  return User(user_id=MAIN_APP_USER_ID, pat=CLARIFAI_PAT)


@pytest.mark.requires_secrets
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
    all_models = list(app.list_models(page_no=1))
    assert len(all_models) == 15  #default per_page is 15

  def test_list_workflows(self, app):
    all_workflows = list(app.list_workflows(page_no=1, per_page=10))
    assert len(all_workflows) == 10

  def test_list_modules(self, app):
    all_modules = list(app.list_modules())
    assert len(all_modules) == 1

  def test_list_installed_module_versions(self, app):
    all_installed_module_versions = list(app.list_installed_module_versions())
    assert len(all_installed_module_versions) == 0

  def test_list_apps(self, client):
    all_apps = list(client.list_apps())
    assert len(all_apps) > 0

  # TODO To resolve `Insufficient scopes` error
  # def test_app_input_count(self, app):
  #   input_count = app.get_input_count()
  #   assert input_count == 41

  def test_get_model(self, client):
    model = client.app(app_id=MAIN_APP_ID).model(model_id=GENERAL_MODEL_ID)
    versions = list(model.list_versions())
    assert len(versions) == 2  #test for list_versions
    assert model.id == GENERAL_MODEL_ID and model.app_id == MAIN_APP_ID and model.user_id == MAIN_APP_USER_ID

  def test_get_workflow(self, client):
    workflow = client.app(app_id=MAIN_APP_ID).workflow(workflow_id=General_Workflow_ID)
    versions = list(workflow.list_versions())
    assert len(versions) == 1  #test for list_versions
    assert workflow.id == General_Workflow_ID and workflow.app_id == MAIN_APP_ID and workflow.user_id == MAIN_APP_USER_ID

  def test_create_app(self):
    app = User(user_id=CREATE_APP_USER_ID, pat=CLARIFAI_PAT).create_app(app_id=CREATE_APP_ID)
    assert app.id == CREATE_APP_ID and app.user_id == CREATE_APP_USER_ID

  def test_create_search(self, create_app):
    search = create_app.search()
    assert search.top_k == DEFAULT_TOP_K and search.metric_distance == "EUCLIDEAN_DISTANCE"

  def test_create_dataset(self, create_app):
    dataset = create_app.create_dataset(CREATE_DATASET_ID)
    assert dataset.id == CREATE_DATASET_ID and dataset.app_id == CREATE_APP_ID and dataset.user_id == CREATE_APP_USER_ID

  def test_create_model(self, create_app):
    model = create_app.create_model(CREATE_MODEL_ID)
    assert model.id == CREATE_MODEL_ID and model.app_id == CREATE_APP_ID and model.user_id == CREATE_APP_USER_ID

  def test_create_module(self, create_app):
    module = create_app.create_module(CREATE_MODULE_ID, description="CI test module")
    assert module.id == CREATE_MODULE_ID and module.app_id == CREATE_APP_ID and module.user_id == CREATE_APP_USER_ID

  def test_create_concept_relations(self, create_app, caplog):
    create_app.create_concepts([OBJECT_CONCEPT_ID, SUBJECT_CONCEPT_ID])
    with caplog.at_level(logging.INFO):
      create_app.create_concept_relations(OBJECT_CONCEPT_ID, [SUBJECT_CONCEPT_ID], [PREDICATE])
      assert "SUCCESS" in caplog.text

  # def test_create_runner(self, client):
  #   client = User(user_id=CREATE_APP_USER_ID, pat=CLARIFAI_PAT)
  #   runner_info = client.create_runner(
  #       CREATE_RUNNER_ID, labels=["ci runner"], description="CI test runner")
  #   assert runner_info.get("runner_id") == CREATE_RUNNER_ID and runner_info.get(
  #       "user_id") == CREATE_APP_USER_ID

  def test_get_dataset(self, create_app):
    dataset = create_app.dataset(dataset_id=CREATE_DATASET_ID)
    dataset.create_version()
    versions = list(dataset.list_versions())
    assert len(versions) == 1  #test for list_versions
    assert dataset.id == CREATE_DATASET_ID and dataset.app_id == CREATE_APP_ID and dataset.user_id == CREATE_APP_USER_ID

  def test_get_module(self, create_app):
    module = create_app.module(module_id=CREATE_MODULE_ID)
    versions = list(module.list_versions())
    assert len(versions) == 0  #test for list_versions
    assert module.id == CREATE_MODULE_ID and module.app_id == CREATE_APP_ID and module.user_id == CREATE_APP_USER_ID

  def test_list_datasets(self, create_app):
    all_datasets = list(create_app.list_datasets())
    assert len(all_datasets) == 1

  def test_search_concept_relations(self, create_app):
    all_concept_relations = list(create_app.search_concept_relations(show_tree=True))
    assert len(all_concept_relations) == 1

  def test_patch_app(self, caplog):
    with caplog.at_level(logging.INFO):
      User(user_id=CREATE_APP_USER_ID).patch_app(
          app_id=CREATE_APP_ID,
          action='overwrite',
          default_language='en',
          base_workflow='Universal',
          description='App Patching Test',
          is_template=True,
          visibility=10,
          notes='App Patching Notes Test',
          image_url=IMAGE_URL)
      assert "SUCCESS" in caplog.text

  def test_patch_dataset(self, create_app, caplog):
    with caplog.at_level(logging.INFO):
      create_app.patch_dataset(
          dataset_id=CREATE_DATASET_ID,
          action='merge',
          description='App Patching Test',
          visibility=10,
          notes='App Patching Notes Test',
          image_url=IMAGE_URL)
      assert "SUCCESS" in caplog.text

  def test_patch_model(self, create_app, caplog):
    with caplog.at_level(logging.INFO):
      create_app.patch_model(
          model_id=CREATE_MODEL_ID,
          action='merge',
          visibility=10,
          description='Model Patching Test',
          notes='Model Patching Test',
          toolkits=['Clarifai'],
          use_cases=['misc'],
          languages=['en'],
          image_url=IMAGE_URL)
      assert "SUCCESS" in caplog.text

  def test_delete_dataset(self, create_app, caplog):
    with caplog.at_level(logging.INFO):
      dataset = create_app.dataset(dataset_id=CREATE_DATASET_ID)
      versions = list(dataset.list_versions())
      dataset.delete_version(version_id=versions[0].version.id)
      assert "SUCCESS" in caplog.text
      create_app.delete_dataset(CREATE_DATASET_ID)
      assert "SUCCESS" in caplog.text

  def test_delete_model(self, create_app, caplog):
    with caplog.at_level(logging.INFO):
      create_app.delete_model(CREATE_MODEL_ID)
      assert "SUCCESS" in caplog.text

  def test_delete_module(self, create_app, caplog):
    with caplog.at_level(logging.INFO):
      create_app.delete_module(CREATE_MODULE_ID)
      assert "SUCCESS" in caplog.text

  def test_delete_concept_relations(self, create_app, caplog):
    with caplog.at_level(logging.INFO):
      all_concept_relation_ids = [
          concept_relation.id for concept_relation in list(create_app.search_concept_relations())
      ]
      create_app.delete_concept_relations(SUBJECT_CONCEPT_ID, all_concept_relation_ids)
      assert "SUCCESS" in caplog.text

  # def test_delete_runner(self, caplog):
  #   client = User(user_id=CREATE_APP_USER_ID)
  #   with caplog.at_level(logging.INFO):
  #     client.delete_runner(CREATE_RUNNER_ID)
  #     assert "SUCCESS" in caplog.text

  def test_delete_app(self, caplog):
    with caplog.at_level(logging.INFO):
      User(user_id=CREATE_APP_USER_ID).delete_app(CREATE_APP_ID)
      assert "SUCCESS" in caplog.text
