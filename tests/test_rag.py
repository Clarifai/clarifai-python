import logging
import os

import pytest

from clarifai.client import User
from clarifai.rag import RAG
from clarifai.urls.helper import ClarifaiUrlHelper

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]

TEXT_FILE_PATH = os.path.dirname(__file__) + "/assets/sample.txt"
PDF_URL = "https://samples.clarifai.com/test_doc.pdf"

CLARIFAI_API_BASE = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")


def client():
    return User(user_id=CREATE_APP_USER_ID, base_url=CLARIFAI_API_BASE)


@pytest.mark.requires_secrets
class TestRAG:
    @classmethod
    def setup_class(self):
        self.rag = RAG.setup(user_id=CREATE_APP_USER_ID, base_url=CLARIFAI_API_BASE)
        wf = self.rag._prompt_workflow
        self.workflow_url = ClarifaiUrlHelper().clarifai_url(
            wf.user_id, wf.app_id, "workflows", wf.id
        )

    def test_setup_correct(self):
        assert len(self.rag._prompt_workflow.workflow_info.nodes) == 2

    def test_from_existing_workflow(self):
        agent = RAG(workflow_url=self.workflow_url)
        assert agent._app.id == self.rag._app.id

    def test_predict_client_manage_state(self):
        messages = [{"role": "human", "content": "What is 1 + 1?"}]
        new_messages = self.rag.chat(messages, client_manage_state=True)
        assert len(new_messages) == 2

    @pytest.mark.skip(reason="Not yet supported. Work in progress.")
    def test_predict_server_manage_state(self):
        messages = [{"role": "human", "content": "What is 1 + 1?"}]
        new_messages = self.rag.chat(messages)
        assert len(new_messages) == 1

    def test_upload_docs_filepath(self, caplog):
        with caplog.at_level(logging.INFO):
            self.rag.upload(file_path=TEXT_FILE_PATH)
            assert "SUCCESS" in caplog.text

    def test_upload_docs_from_url(self, caplog):
        with caplog.at_level(logging.INFO):
            self.rag.upload(url=PDF_URL)
            assert "SUCCESS" in caplog.text

    @classmethod
    def teardown_class(self):
        client().delete_app(self.rag._app.id)
