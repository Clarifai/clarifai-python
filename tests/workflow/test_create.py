import glob
import logging
import os
import typing

import pytest

from clarifai.client.user import User

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]
CREATE_APP_ID = "ci_workflow_create"


def get_test_parse_workflow_creation_workflows() -> typing.List[str]:
  filenames = glob.glob("tests/workflow/fixtures/*.yml")
  return filenames


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
      self.app.create_workflow(filename, generate_new_id=True)
      assert "Workflow created" in caplog.text

  @classmethod
  def teardown_class(cls):
    cls.client.delete_app(app_id=CREATE_APP_ID)
