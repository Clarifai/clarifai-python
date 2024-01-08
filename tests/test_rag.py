import os
from collections import namedtuple

import pytest

from clarifai.rag import RAG
from clarifai.urls.helper import ClarifaiUrlHelper

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]

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
