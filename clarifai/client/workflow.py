import os
import time
from typing import Dict, Generator, List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Input
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.client.base import BaseClient
from clarifai.client.input import Inputs
from clarifai.client.lister import Lister
from clarifai.client.model import Model
from clarifai.constants.workflow import MAX_WORKFLOW_PREDICT_INPUTS
from clarifai.errors import UserError
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.logging import logger
from clarifai.utils.misc import BackoffIterator
from clarifai.workflows.export import Exporter


class Workflow(Lister, BaseClient):
  """Workflow is a class that provides access to Clarifai API endpoints related to Workflow information."""

  def __init__(self,
               url: str = None,
               workflow_id: str = None,
               workflow_version: Dict = {'id': ""},
               output_config: Dict = {'min_value': 0},
               base_url: str = "https://api.clarifai.com",
               pat: str = None,
               token: str = None,
               root_certificates_path: str = None,
               **kwargs):
    """Initializes a Workflow object.

    Args:
        url (str): The URL to initialize the workflow object.
        workflow_id (str): The Workflow ID to interact with.
        workflow_version (dict): The Workflow Version to interact with.
        output_config (dict): The output config to interact with.
          min_value (float): The minimum value of the prediction confidence to filter.
          max_concepts (int): The maximum number of concepts to return.
          select_concepts (list[Concept]): The concepts to select.
          sample_ms (int): The number of milliseconds to sample.
        base_url (str): Base API url. Default "https://api.clarifai.com"
        pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
        token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
        root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
        **kwargs: Additional keyword arguments to be passed to the Workflow.
    """
    if url and workflow_id:
      raise UserError("You can only specify one of url or workflow_id.")
    if not url and not workflow_id:
      raise UserError("You must specify one of url or workflow_id.")
    if url:
      user_id, app_id, _, workflow_id, workflow_version_id = ClarifaiUrlHelper.split_clarifai_url(
          url)
      workflow_version = {'id': workflow_version_id}
      kwargs = {'user_id': user_id, 'app_id': app_id}
    self.kwargs = {**kwargs, 'id': workflow_id, 'version': workflow_version}
    self.output_config = output_config
    self.workflow_info = resources_pb2.Workflow(**self.kwargs)
    self.logger = logger
    self.input_types = None
    BaseClient.__init__(
        self,
        user_id=self.user_id,
        app_id=self.app_id,
        base=base_url,
        pat=pat,
        token=token,
        root_certificates_path=root_certificates_path)
    Lister.__init__(self)

  def predict(self, inputs: List[Input], workflow_state_id: str = None):
    """Predicts the workflow based on the given inputs.

    Args:
        inputs (list[Input]): The inputs to predict.
        workflow_state_id (str): key for the workflow state cache that stores the workflow node predictions.
    """
    if len(inputs) > MAX_WORKFLOW_PREDICT_INPUTS:
      raise UserError(f"Too many inputs. Max is {MAX_WORKFLOW_PREDICT_INPUTS}."
                     )  # TODO Use Chunker for inputs len > 32
    request = service_pb2.PostWorkflowResultsRequest(
        user_app_id=self.user_app_id,
        workflow_id=self.id,
        version_id=self.version.id,
        inputs=inputs,
        output_config=self.output_config)

    if workflow_state_id:
      request.workflow_state.id = workflow_state_id

    start_time = time.time()
    backoff_iterator = BackoffIterator(10)

    while True:
      response = self._grpc_request(self.STUB.PostWorkflowResults, request)

      if response.status.code == status_code_pb2.MODEL_DEPLOYING and \
          time.time() - start_time < 60*10:  # 10 minutes
        self.logger.info(f"{self.id} Workflow is still deploying, please wait...")
        time.sleep(next(backoff_iterator))
        continue

      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Workflow Predict failed with response {response.status!r}")
      else:
        break

    return response

  def predict_by_filepath(self, filepath: str, input_type: str = None):
    """Predicts the workflow based on the given filepath.

    Args:
        filepath (str): The filepath to predict.
        input_type (str, optional): The type of input. Can be 'image', 'text', 'video' or 'audio.

    Example:
        >>> from clarifai.client.workflow import Workflow
        >>> workflow = Workflow("url") # Example: https://clarifai.com/clarifai/main/workflows/Face-Sentiment
                      or
        >>> workflow = Workflow(user_id='user_id', app_id='app_id', workflow_id='workflow_id')
        >>> workflow_prediction = workflow.predict_by_filepath('filepath')
    """
    if not input_type:
      self.load_info()
      if len(self.input_types) > 1:
        raise UserError("Workflow has multiple input types. Please use workflow.predict().")
      input_type = self.input_types[0]

    if input_type not in {'image', 'text', 'video', 'audio'}:
      raise UserError('Invalid input type it should be image, text, video or audio.')
    if not os.path.isfile(filepath):
      raise UserError('Invalid filepath.')

    with open(filepath, "rb") as f:
      file_bytes = f.read()

    return self.predict_by_bytes(file_bytes, input_type)

  def predict_by_bytes(self, input_bytes: bytes, input_type: str = None):
    """Predicts the workflow based on the given bytes.

    Args:
        input_bytes (bytes): Bytes to predict on.
        input_type (str, optional): The type of input. Can be 'image', 'text', 'video' or 'audio.
    """
    if not input_type:
      self.load_info()
      if len(self.input_types) > 1:
        raise UserError("Workflow has multiple input types. Please use workflow.predict().")
      input_type = self.input_types[0]

    if input_type not in {'image', 'text', 'video', 'audio'}:
      raise UserError('Invalid input type it should be image, text, video or audio.')
    if not isinstance(input_bytes, bytes):
      raise UserError('Invalid bytes.')

    if input_type == "image":
      input_proto = Inputs.get_input_from_bytes("", image_bytes=input_bytes)
    elif input_type == "text":
      input_proto = Inputs.get_input_from_bytes("", text_bytes=input_bytes)
    elif input_type == "video":
      input_proto = Inputs.get_input_from_bytes("", video_bytes=input_bytes)
    elif input_type == "audio":
      input_proto = Inputs.get_input_from_bytes("", audio_bytes=input_bytes)

    return self.predict(inputs=[input_proto])

  def predict_by_url(self, url: str, input_type: str = None):
    """Predicts the workflow based on the given URL.

    Args:
        url (str): The URL to predict.
        input_type (str, optional): The type of input. Can be 'image', 'text', 'video' or 'audio.

    Example:
        >>> from clarifai.client.workflow import Workflow
        >>> workflow = Workflow("url") # Example: https://clarifai.com/clarifai/main/workflows/Face-Sentiment
                      or
        >>> workflow = Workflow(user_id='user_id', app_id='app_id', workflow_id='workflow_id')
        >>> workflow_prediction = workflow.predict_by_url('url')
    """
    if not input_type:
      self.load_info()
      if len(self.input_types) > 1:
        raise UserError("Workflow has multiple input types. Please use workflow.predict().")
      input_type = self.input_types[0]

    if input_type not in {'image', 'text', 'video', 'audio'}:
      raise UserError('Invalid input type it should be image, text, video or audio.')

    if input_type == "image":
      input_proto = Inputs.get_input_from_url("", image_url=url)
    elif input_type == "text":
      input_proto = Inputs.get_input_from_url("", text_url=url)
    elif input_type == "video":
      input_proto = Inputs.get_input_from_url("", video_url=url)
    elif input_type == "audio":
      input_proto = Inputs.get_input_from_url("", audio_url=url)

    return self.predict(inputs=[input_proto])

  def list_versions(self, page_no: int = None,
                    per_page: int = None) -> Generator['Workflow', None, None]:
    """Lists all the versions of the workflow.

    Args:
        page_no (int): The page number to list.
        per_page (int): The number of items per page.

    Yields:
        Workflow: Workflow objects for versions of the workflow.

    Example:
        >>> from clarifai.client.workflow import Workflow
        >>> workflow = Workflow(user_id='user_id', app_id='app_id', workflow_id='workflow_id')
        >>> workflow_versions = list(workflow.list_versions())

    Note:
        Defaults to 16 per page if page_no is specified and per_page is not specified.
        If both page_no and per_page are None, then lists all the resources.
    """
    request_data = dict(
        user_app_id=self.user_app_id,
        workflow_id=self.id,
    )
    all_workflow_versions_info = self.list_pages_generator(
        self.STUB.ListWorkflowVersions,
        service_pb2.ListWorkflowVersionsRequest,
        request_data,
        per_page=per_page,
        page_no=page_no)

    for workflow_version_info in all_workflow_versions_info:
      workflow_version_info['id'] = workflow_version_info['workflow_version_id']
      del workflow_version_info['workflow_version_id']
      yield Workflow.from_auth_helper(
          auth=self.auth_helper,
          workflow_id=self.id,
          **dict(self.kwargs, version=workflow_version_info))

  def export(self, out_path: str):
    """Exports the workflow to a yaml file.

    Args:
        out_path (str): The path to save the yaml file to.

    Example:
        >>> from clarifai.client.workflow import Workflow
        >>> workflow = Workflow("https://clarifai.com/clarifai/main/workflows/Demographics")
        >>> workflow.export('out_path.yml')
    """
    request = service_pb2.GetWorkflowRequest(user_app_id=self.user_app_id, workflow_id=self.id)
    response = self._grpc_request(self.STUB.GetWorkflow, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(f"Workflow Export failed with response {response.status!r}")

    with Exporter(response) as e:
      e.parse()
      e.export(out_path)

    self.logger.info(f"Exported workflow to {out_path}")

  def load_info(self) -> None:
    """Loads the workflow info."""
    if not self.input_types:
      request = service_pb2.GetWorkflowRequest(user_app_id=self.user_app_id, workflow_id=self.id)
      response = self._grpc_request(self.STUB.GetWorkflow, request)
      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Workflow Get failed with response {response.status!r}")

      dict_response = MessageToDict(response, preserving_proto_field_name=True)
      self.kwargs = self.process_response_keys(dict_response['workflow'])
      self.workflow_info = resources_pb2.Workflow(**self.kwargs)

      model = Model(
          model_id=self.kwargs['nodes'][0]['model']['id'],
          **self.kwargs['nodes'][0]['model'],
          pat=self.pat)
      model.load_input_types()
      self.input_types = model.input_types

  def __getattr__(self, name):
    return getattr(self.workflow_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.workflow_info, param)}" for param in init_params
        if hasattr(self.workflow_info, param)
    ]
    return f"Workflow Details: \n{', '.join(attribute_strings)}\n"
