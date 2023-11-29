import glob
import logging
import os
import typing
from datetime import datetime

import pytest

from clarifai.client.user import User

NOW = str(int(datetime.now().timestamp()))
CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]
CREATE_APP_ID = f"test_workflow_create_delete_app_{NOW}"


def get_test_parse_workflow_creation_workflows() -> typing.List[str]:
  filenames = glob.glob("tests/workflow/fixtures/*.yml")
  return filenames


@pytest.mark.requires_secrets
class TestWorkflowCreate:

  @classmethod
  def setup_class(cls):
    cls.client = User(user_id=CREATE_APP_USER_ID)
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

  def test_delete_workflow(self, caplog):
    with caplog.at_level(logging.INFO):
      self.app.delete_workflow("General")
      assert "Workflow Deleted" in caplog.text

  @classmethod
  def teardown_class(cls):
    cls.client.delete_app(app_id=CREATE_APP_ID)
