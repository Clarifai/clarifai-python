import uuid
from datetime import datetime
from typing import List

import yaml
from clarifai_grpc.grpc.api import resources_pb2  # noqa: F401
from google.protobuf.struct_pb2 import Struct

from clarifai.client.app import App
from clarifai.client.input import Inputs
from clarifai.client.model import Model
from clarifai.client.user import User
from clarifai.client.workflow import Workflow
from clarifai.constants.rag import MAX_UPLOAD_BATCH_SIZE
from clarifai.errors import UserError
from clarifai.rag.utils import (convert_messages_to_str, format_assistant_message, load_documents,
                                split_document)
from clarifai.utils.logging import get_logger


class RAG:
  """
    RAG is a class for Retrieval Augmented Generation.

    Example:
        >>> from clarifai.rag import RAG
        >>> rag_agent = RAG()
    """
  chat_state_id = None

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
    if not user_id:
      raise UserError(
          "user_id must be provided. It can be found at https://clarifai.com/settings.")
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

  def upload(self,
             file_path: str = None,
             folder_path: str = None,
             url: str = None,
             batch_size: int = 128,
             chunk_size: int = 1024,
             chunk_overlap: int = 200,
             **kwargs) -> None:
    """Uploads documents to the app.
        - Read from a local directory or public url or local filename.
        - Parse the document(s) into chunks.
        - Ingest chunks into the app with metadata.

    Args:
        file_path str: File path to the document.
        folder_path str: Folder path to the documents.
        url str: Public url to the document.
        batch_size int: Batch size for uploading.
        chunk_size int: Chunk size for splitting the document.
        chunk_overlap int: The token overlap of each chunk when splitting.
        **kwargs: Additional arguments for the SentenceSplitter. Refer https://docs.llamaindex.ai/en/stable/api/llama_index.node_parser.SentenceSplitter.html

    Example:
        >>> from clarifai.rag import RAG
        >>> rag_agent = RAG.setup()
        >>> rag_agent.upload(folder_path = "~/work/docs")
        >>> rag_agent.upload(file_path = "~/work/docs/manual.pdf")
    """
    #set batch size
    if batch_size > MAX_UPLOAD_BATCH_SIZE:
      raise ValueError(f"batch_size cannot be greater than {MAX_UPLOAD_BATCH_SIZE}")

    #check if only one of file_path, folder_path, or url is specified
    if file_path and (folder_path or url):
      raise ValueError("Only one of file_path, folder_path, or url can be specified.")
    if folder_path and (file_path or url):
      raise ValueError("Only one of file_path, folder_path, or url can be specified.")
    if url and (file_path or folder_path):
      raise ValueError("Only one of file_path, folder_path, or url can be specified.")

    #loading documents
    documents = load_documents(file_path=file_path, folder_path=folder_path, url=url)

    #splitting documents into chunks
    text_chunks = []
    metadata = []

    #iterate through documents
    for doc in documents:
      cur_text_chunks = split_document(
          text=doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
      text_chunks.extend(cur_text_chunks)
      metadata.extend([doc.metadata for _ in range(len(cur_text_chunks))])
      #if batch size is reached, upload the batch
      if len(text_chunks) > batch_size:
        for idx in range(0, len(text_chunks), batch_size):
          if idx + batch_size > len(text_chunks):
            continue
          batch_texts = text_chunks[0:batch_size]
          batch_ids = [uuid.uuid4().hex for _ in range(batch_size)]
          #metadata
          batch_metadatas = metadata[0:batch_size]
          meta_list = []
          for meta in batch_metadatas:
            meta_struct = Struct()
            meta_struct.update(meta)
            meta_list.append(meta_struct)
          del batch_metadatas
          #creating input proto
          input_batch = [
              self._app.inputs().get_text_input(
                  input_id=batch_ids[i],
                  raw_text=text,
                  metadata=meta_list[i],
              ) for i, text in enumerate(batch_texts)
          ]
          #uploading input with metadata
          self._app.inputs().upload_inputs(inputs=input_batch)
          #delete uploaded chunks
          del text_chunks[0:batch_size]
          del metadata[0:batch_size]

    #uploading the remaining chunks
    if len(text_chunks) > 0:
      batch_size = len(text_chunks)
      batch_ids = [uuid.uuid4().hex for _ in range(batch_size)]
      #metadata
      batch_metadatas = metadata[0:batch_size]
      meta_list = []
      for meta in batch_metadatas:
        meta_struct = Struct()
        meta_struct.update(meta)
        meta_list.append(meta_struct)
      del batch_metadatas
      #creating input proto
      input_batch = [
          self._app.inputs().get_text_input(
              input_id=batch_ids[i],
              raw_text=text,
              metadata=meta_list[i],
          ) for i, text in enumerate(text_chunks)
      ]
      #uploading input with metadata
      self._app.inputs().upload_inputs(inputs=input_batch)
      del text_chunks
      del metadata

  def chat(self, messages: List[dict], client_manage_state: bool = False) -> List[dict]:
    """Chat interface in OpenAI API format.

    Args:
        messages List[dict]: A list of dictionary in the following format:
        ```
        [
          {"role": "user", "content": "Hello there."},
          {"role": "assistant", "content": "Hi, I'm Claude. How can I help you?"},
          {"role": "user", "content": "Can you explain LLMs in plain English?"},
        ]
        ```
        client_manage_state (bool): Whether the client will handle chat state management. Default is false.

    This will pass back the workflow state ID for the server to store chat state.
    """
    if client_manage_state:
      single_prompt = convert_messages_to_str(messages)
      input_proto = Inputs._get_proto("", "", text_pb=resources_pb2.Text(raw=single_prompt))
      response = self._prompt_workflow.predict([input_proto])
      messages.append(format_assistant_message(response.results[0].outputs[-1].data.text.raw))
      return messages

    # server-side state management
    message = messages[-1].get("content", "")
    if len(message) == 0:
      raise UserError("Empty message supplied.")

    # get chat state id
    chat_state_id = "init" if self.chat_state_id is None else self.chat_state_id

    # call predict
    input_proto = Inputs._get_proto("", "", text_pb=resources_pb2.Text(raw=message))
    response = self._prompt_workflow.predict([input_proto], workflow_state_id=chat_state_id)

    # store chat state id
    self.chat_state_id = response.workflow_state.id
    return [format_assistant_message(response.results[0].outputs[-1].data.text.raw)]
