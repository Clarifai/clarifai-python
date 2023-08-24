import os
from typing import Dict, List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Input
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.errors import UserError
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.logging import get_logger


class Workflow(Lister, BaseClient):
  """Workflow is a class that provides access to Clarifai API endpoints related to Workflow information."""

  def __init__(self,
               url_init: str = "",
               workflow_id: str = "",
               workflow_version: Dict = {'id': ""},
               output_config: Dict = {'min_value': 0},
               **kwargs):
    """Initializes a Workflow object.

    Args:
        url_init (str): The URL to initialize the workflow object.
        workflow_id (str): The Workflow ID to interact with.
        workflow_version (dict): The Workflow Version to interact with.
        output_config (dict): The output config to interact with.
          min_value (float): The minimum value of the prediction confidence to filter.
          max_concepts (int): The maximum number of concepts to return.
          select_concepts (list[Concept]): The concepts to select.
          sample_ms (int): The number of milliseconds to sample.
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
    """
    if url_init != "" and workflow_id != "":
      raise UserError("You can only specify one of url_init or workflow_id.")
    if url_init == "" and workflow_id == "":
      raise UserError("You must specify one of url_init or workflow_id.")
    if url_init != "":
      user_id, app_id, _, workflow_id, workflow_version_id = ClarifaiUrlHelper.split_clarifai_url(
          url_init)
      workflow_version = {'id': workflow_version_id}
      kwargs = {'user_id': user_id, 'app_id': app_id}
    self.kwargs = {**kwargs, 'id': workflow_id, 'version': workflow_version}
    self.output_config = output_config
    self.workflow_info = resources_pb2.Workflow(**self.kwargs)
    self.logger = get_logger(logger_level="INFO")
    BaseClient.__init__(self, user_id=self.user_id, app_id=self.app_id)
    Lister.__init__(self)

  def predict(self, inputs: List[Input]):
    """Predicts the workflow based on the given inputs.

    Args:
        inputs (list[Input]): The inputs to predict.
    """
    if len(inputs) > 128:
      raise UserError("Too many inputs. Max is 128.")  # TODO Use Chunker for inputs len > 128
    request = service_pb2.PostWorkflowResultsRequest(
        user_app_id=self.user_app_id,
        workflow_id=self.id,
        version_id=self.version.id,
        inputs=inputs,
        output_config=self.output_config)

    response = self._grpc_request(self.STUB.PostWorkflowResults, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(f"Workflow Predict failed with response {response.status!r}")

    return response

  def predict_by_filepath(self, filepath: str, input_type: str):
    """Predicts the workflow based on the given filepath.

    Args:
        filepath (str): The filepath to predict.
        input_type (str): The type of input. Can be 'image', 'text', 'video' or 'audio.

    Example:
        >>> from clarifai.client.workflow import Workflow
        >>> workflow = Workflow("workflow_url") # Example: https://clarifai.com/clarifai/main/workflows/Face-Sentiment
                      or
        >>> workflow = Workflow(user_id='user_id', app_id='app_id', workflow_id='workflow_id')
        >>> workflow_prediction = workflow.predict_by_filepath('filepath', 'image')
    """
    if input_type not in {'image', 'text', 'video', 'audio'}:
      raise UserError('Invalid input type it should be image, text, video or audio.')
    if not os.path.isfile(filepath):
      raise UserError('Invalid filepath.')

    with open(filepath, "rb") as f:
      file_bytes = f.read()

    return self.predict_by_bytes(file_bytes, input_type)

  def predict_by_bytes(self, input_bytes: bytes, input_type: str):
    """Predicts the workflow based on the given bytes.

    Args:
        input_bytes (bytes): Bytes to predict on.
        input_type (str): The type of input. Can be 'image', 'text', 'video' or 'audio.
    """
    if input_type not in {'image', 'text', 'video', 'audio'}:
      raise UserError('Invalid input type it should be image, text, video or audio.')
    if not isinstance(input_bytes, bytes):
      raise UserError('Invalid bytes.')

    if input_type == "image":
      input_proto = resources_pb2.Input(
          data=resources_pb2.Data(image=resources_pb2.Image(base64=input_bytes)))
    elif input_type == "text":
      input_proto = resources_pb2.Input(
          data=resources_pb2.Data(text=resources_pb2.Text(raw=input_bytes)))
    elif input_type == "video":
      input_proto = resources_pb2.Input(
          data=resources_pb2.Data(video=resources_pb2.Video(base64=input_bytes)))
    elif input_type == "audio":
      input_proto = resources_pb2.Input(
          data=resources_pb2.Data(audio=resources_pb2.Audio(base64=input_bytes)))

    return self.predict(inputs=[input_proto])

  def predict_by_url(self, url: str, input_type: str):
    """Predicts the workflow based on the given URL.

    Args:
        url (str): The URL to predict.
        input_type (str): The type of input. Can be 'image', 'text', 'video' or 'audio.

    Example:
        >>> from clarifai.client.workflow import Workflow
        >>> workflow = Workflow("workflow_url") # Example: https://clarifai.com/clarifai/main/workflows/Face-Sentiment
                      or
        >>> workflow = Workflow(user_id='user_id', app_id='app_id', workflow_id='workflow_id')
        >>> workflow_prediction = workflow.predict_by_url('url', 'image')
    """
    if input_type not in {'image', 'text', 'video', 'audio'}:
      raise UserError('Invalid input type it should be image, text, video or audio.')

    if input_type == "image":
      input_proto = resources_pb2.Input(
          data=resources_pb2.Data(image=resources_pb2.Image(url=url)))
    elif input_type == "text":
      input_proto = resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(url=url)))
    elif input_type == "video":
      input_proto = resources_pb2.Input(
          data=resources_pb2.Data(video=resources_pb2.Video(url=url)))
    elif input_type == "audio":
      input_proto = resources_pb2.Input(
          data=resources_pb2.Data(audio=resources_pb2.Audio(url=url)))

    return self.predict(inputs=[input_proto])

  def list_versions(self) -> List['Workflow']:
    """Lists all the versions of the workflow.

    Returns:
        list[Workflow]: A list of Workflow objects.

    Example:
        >>> from clarifai.client.workflow import Workflow
        >>> workflow = Workflow(user_id='user_id', app_id='app_id', workflow_id='workflow_id')
        >>> workflow_versions = workflow.list_versions()
    """
    request_data = dict(
        user_app_id=self.user_app_id,
        workflow_id=self.id,
        per_page=self.default_page_size,
    )
    all_workflow_versions_info = list(
        self.list_all_pages_generator(self.STUB.ListWorkflowVersions,
                                      service_pb2.ListWorkflowVersionsRequest, request_data))

    for workflow_version_info in all_workflow_versions_info:
      workflow_version_info['id'] = workflow_version_info['workflow_version_id']
      del workflow_version_info['workflow_version_id']

    return [
        Workflow(workflow_id=self.id, **dict(self.kwargs, version=workflow_version_info))
        for workflow_version_info in all_workflow_versions_info
    ]

  def __getattr__(self, name):
    return getattr(self.workflow_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.workflow_info, param)}" for param in init_params
        if hasattr(self.workflow_info, param)
    ]
    return f"Workflow Details: \n{', '.join(attribute_strings)}\n"
