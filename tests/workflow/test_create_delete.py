import glob
import logging
import os
import typing
import uuid

import pytest

from clarifai.client.user import User

NOW = uuid.uuid4().hex[:10]
CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]
CREATE_APP_ID = f"test_workflow_create_delete_app_{NOW}"

#assets
IMAGE_URL = "https://samples.clarifai.com/metro-north.jpg"

CLARIFAI_API_BASE = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")


def get_test_parse_workflow_creation_workflows() -> typing.List[str]:
  filenames = glob.glob("tests/workflow/fixtures/*.yml")
  return filenames


@pytest.mark.requires_secrets
class TestWorkflowCreate:

  @classmethod
  def setup_class(cls):
    cls.client = User(user_id=CREATE_APP_USER_ID, base_url=CLARIFAI_API_BASE)
    try:
      cls.app = cls.client.create_app(app_id=CREATE_APP_ID, base_workflow="Empty")
    except Exception as e:
      if "already exists" in str(e):
        cls.app = cls.client.app(app_id=CREATE_APP_ID)

  @pytest.mark.parametrize("filename", get_test_parse_workflow_creation_workflows())
  def test_parse_workflow_creation(self, filename: str, caplog):
    with caplog.at_level(logging.INFO):
      if "general.yml" in filename:
        generate_new_id = False
      else:
        generate_new_id = True
      self.app.create_workflow(filename, generate_new_id=generate_new_id)
      assert "Workflow created" in caplog.text

  def test_patch_workflow(self, caplog):
    with caplog.at_level(logging.INFO):
      workflow_id = list(self.app.list_workflows())[0].id
      self.app.patch_workflow(
          workflow_id=workflow_id,
          config_filepath='tests/workflow/fixtures/general.yml',
          visibility=10,
          description='Workflow Patching Test',
          notes='Workflow Patching Test',
          image_url=IMAGE_URL)
      assert "Workflow patched" in caplog.text

  def test_delete_workflow(self, caplog):
    with caplog.at_level(logging.INFO):
      self.app.delete_workflow("General")
      assert "Workflow Deleted" in caplog.text

  @classmethod
  def teardown_class(cls):
    cls.client.delete_app(app_id=CREATE_APP_ID)
