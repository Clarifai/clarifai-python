import os
from collections import namedtuple

import pytest

from clarifai.rag import RAG
from clarifai.urls.helper import ClarifaiUrlHelper

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]

auth_obj = namedtuple("auth", "ui")


@pytest.mark.requires_secrets
class TestRAG:
  workflow_url = ""
  app_id = ""

  def test_setup(self):
    app = RAG.setup(user_id=CREATE_APP_USER_ID)
    wf = app._prompt_workflow
    assert len(wf.workflow_info.nodes) == 2
    auth = auth_obj(ui="https://clarifai.com")
    self.workflow_url = ClarifaiUrlHelper(auth).clarifai_url(wf.user_id, wf.app_id, "workflows",
                                                             wf.id)
    self.app_id = app._app.id

    ## test_from_existing_workflow
    app = RAG(workflow_url=self.workflow_url)
    assert app._app.id == self.app_id
