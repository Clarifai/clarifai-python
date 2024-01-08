from datetime import datetime

import yaml
from google.protobuf.struct_pb2 import Struct

from clarifai.client.app import App
from clarifai.client.model import Model
from clarifai.client.user import User
from clarifai.client.workflow import Workflow
from clarifai.utils.logging import get_logger


class RAG:
  """
    RAG is a class for Retrieval Augmented Generation.

    Example:
        >>> from clarifai.rag import RAG
        >>> rag_agent = RAG()
    """

  def __init__(self,
               workflow_url: str = None,
               workflow: Workflow = None,
               base_url: str = "https://api.clarifai.com",
               pat: str = None,
               **kwargs):
    """Initialize an empty or existing RAG.
    """
    self.logger = get_logger(logger_level="INFO", name=__name__)
    if workflow_url is not None and workflow is None:
      self.logger.info("workflow_url:%s", workflow_url)
      w = Workflow(workflow_url, base_url=base_url, pat=pat)
      self._prompt_workflow = w
      self._app = App(app_id=w.app_id, base_url=w.base, pat=w.pat)
    elif workflow_url is None and workflow is not None:
      self._prompt_workflow = workflow
      self._app = App(app_id=workflow.app_id, base_url=workflow.base, pat=workflow.pat)

  @classmethod
  def setup(cls,
            user_id: str = None,
            llm_url: str = "https://clarifai.com/mistralai/completion/models/mistral-7B-Instruct",
            base_workflow: str = "Text",
            workflow_yaml_filename: str = 'prompter_wf.yaml',
            base_url: str = "https://api.clarifai.com",
            pat: str = None,
            **kwargs):
    """Creates an app with `Text` as base workflow, create prompt model, create prompt workflow.

    Example:
        >>> from clarifai.rag import RAG
        >>> rag_agent = RAG.setup()
    """
    user = User(user_id=user_id, base_url=base_url, pat=pat)
    llm = Model(llm_url)

    params = Struct()
    params.update({
        "prompt_template":
            "Context information is below:\n{data.hits}\nGiven the context information and not prior knowledge, answer the query.\nQuery: {data.text.raw}\nAnswer: "
    })
    prompter_model_params = {"params": params}

    ## Create an App
    now_ts = str(int(datetime.now().timestamp()))
    app_id = f"rag_app_{now_ts}"
    app = user.create_app(app_id=app_id, base_workflow=base_workflow)

    ## Create rag-prompter model and version
    prompter_model = app.create_model(
        model_id=f"rag_prompter_{now_ts}", model_type_id="rag-prompter")
    prompter_model = prompter_model.create_version(output_info=prompter_model_params)

    ## Generate a tmp yaml file for workflow creation
    workflow_id = f"rag-wf-{now_ts}"
    workflow_dict = {
        "workflow": {
            "id":
                workflow_id,
            "nodes": [{
                "id": "rag-prompter",
                "model": {
                    "model_id": prompter_model.id,
                    "model_version_id": prompter_model.model_version.id
                }
            }, {
                "id": "llm",
                "model": {
                    "model_id": llm.id,
                    "user_id": llm.user_id,
                    "app_id": llm.app_id
                },
                "node_inputs": [{
                    "node_id": "rag-prompter"
                }]
            }]
        }
    }
    with open(workflow_yaml_filename, 'w') as out_file:
      yaml.dump(workflow_dict, out_file, default_flow_style=False)

    ## Create prompt workflow
    wf = app.create_workflow(config_filepath=workflow_yaml_filename)
    del user, llm, prompter_model, prompter_model_params
    return cls(workflow=wf)

  # TODO: Implement this.
  def upload():
    """Does the following:
        - Read from a local directory or public url or local filename.
        - Parse the document(s) into chunks.
        - Ingest chunks into the app with metadata.

    Example:
        >>> from clarifai.rag import RAG
        >>> rag_agent = RAG().setup()
        >>> rag_agent.upload("~/work/docs")
        >>> rag_agent.upload("~/work/docs/manual.pdf")
    """
    pass

  # TODO: Implement this.
  def chat(message: str) -> str:
    """Call self._prompt_workflow.predict_by_bytes.

    This will pass back the workflow state ID for the server to store chat state.
    """
    pass
