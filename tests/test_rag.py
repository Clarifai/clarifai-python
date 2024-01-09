import os
import logging
from collections import namedtuple

import pytest

from clarifai.rag import RAG
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.client import User

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]

TEXT_FILE_PATH = os.path.dirname(__file__) + "/assets/sample.txt"

auth_obj = namedtuple("auth", "ui")


@pytest.mark.requires_secrets
class TestRAG:

  @classmethod
  def setup_class(self):
    self.rag = RAG.setup(user_id=CREATE_APP_USER_ID)
    wf = self.rag._prompt_workflow
    auth = auth_obj(ui="https://clarifai.com")
    self.workflow_url = ClarifaiUrlHelper(auth).clarifai_url(wf.user_id, wf.app_id, "workflows",
                                                             wf.id)

  def test_setup_correct(self):
    assert len(self.rag._prompt_workflow.workflow_info.nodes) == 2

  def test_from_existing_workflow(self):
    app = RAG(workflow_url=self.workflow_url)
    assert app._app.id == self.rag._app.id

  def test_predict_client_manage_state(self):
    messages = [{"role": "human", "content": "What is 1 + 1?"}]
    new_messages = self.rag.chat(messages, client_manage_state=True)
    assert len(new_messages) == 2

  @pytest.mark.skip(reason="Not yet supported. Work in progress.")
  def test_predict_server_manage_state(self):
    messages = [{"role": "human", "content": "What is 1 + 1?"}]
    new_messages = self.rag.chat(messages)
    assert len(new_messages) == 1

  def test_upload_docs(self, caplog):
    with caplog.at_level(logging.INFO):
      self.rag.upload(file_path=TEXT_FILE_PATH)
      assert "SUCCESS" in caplog.text

  @classmethod
  def teardown_class(self):
    User(user_id=CREATE_APP_USER_ID).delete_app(self.rag._app.id)
