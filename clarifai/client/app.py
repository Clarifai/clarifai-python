from typing import List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.client.base import BaseClient
from clarifai.client.dataset import Dataset
from clarifai.client.lister import Lister
from clarifai.client.model import Model
from clarifai.client.workflow import Workflow
from clarifai.utils.logging import get_logger


class App(Lister, BaseClient):
  """
  App is a class that provides access to Clarifai API endpoints related to App information.
  Inherits from BaseClient for authentication purposes.
  """

  def __init__(self, app_id: str, **kwargs):
    """Initializes an App object.
    Args:
        app_id (str): The App ID for the App to interact with.
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
            - name (str): The name of the app.
            - description (str): The description of the app.
    """
    self.kwargs = {**kwargs, 'id': app_id}
    self.app_info = resources_pb2.App(**self.kwargs)
    self.logger = get_logger(logger_level="INFO", name=__name__)
    BaseClient.__init__(self, user_id=self.user_id, app_id=self.id)
    Lister.__init__(self)

  def list_datasets(self) -> List[Dataset]:
    """Lists all the datasets for the app."""
    request_data = dict(
        user_app_id=self.user_app_id,
        per_page=self.default_page_size,
    )
    all_datasets_info = list(
        self.list_all_pages_generator(self.STUB.ListDatasets, service_pb2.ListDatasetsRequest,
                                      request_data))
    for dataset_info in all_datasets_info:
      if 'version' in list(dataset_info.keys()):
        del dataset_info['version']['metrics']

    return [Dataset(**dataset_info) for dataset_info in all_datasets_info]

  def list_models(self):
    """
    Lists all the models for the app.
    """
    pass  # TODO

  def list_workflows(self):
    """
    Lists all the workflows for the app.
    """
    pass  # TODO

  def list_concepts(self):
    """
    Lists all the concepts for the app.
    """
    pass  # TODO

  def create_dataset(self, dataset_id: str, **kwargs) -> Dataset:
    """Creates a dataset for the app.
    Args:
        dataset_id (str): The dataset ID for the dataset to create.
        **kwargs: Additional keyword arguments to be passed to the Dataset.
    Returns:
        Dataset: A Dataset object for the specified dataset ID.
    """
    request = service_pb2.PostDatasetsRequest(
        user_app_id=self.user_app_id, datasets=[resources_pb2.Dataset(id=dataset_id, **kwargs)])
    response = self._grpc_request(self.STUB.PostDatasets, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nDataset created\n%s", response.status)

    return Dataset(dataset_id=dataset_id, **kwargs)

  def create_model(self, model_id: str, **kwargs) -> Model:
    """Creates a model for the app.
    Args:
        model_id (str): The model ID for the model to create.
        **kwargs: Additional keyword arguments to be passed to the Model.
    Returns:
        Model: A Model object for the specified model ID.
    """
    request = service_pb2.PostModelsRequest(
        user_app_id=self.user_app_id, models=[resources_pb2.Model(id=model_id, **kwargs)])
    response = self._grpc_request(self.STUB.PostModels, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nModel created\n%s", response.status)

    return Model(model_id=model_id, **kwargs)

  def create_workflow(self, workflow_id: str, **kwargs) -> Workflow:
    """Creates a workflow for the app.
    Args:
        workflow_id (str): The workflow ID for the workflow to create.
        **kwargs: Additional keyword arguments to be passed to the workflow.
    Returns:
        Workflow: A Workflow object for the specified workflow ID.
    """
    request = service_pb2.PostWorkflowsRequest(
        user_app_id=self.user_app_id, workflows=[resources_pb2.Workflow(id=workflow_id, **kwargs)])
    response = self._grpc_request(self.STUB.PostWorkflows, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nWorkflow created\n%s", response.status)

    return Workflow(workflow_id=workflow_id, **kwargs)

  def dataset(self, dataset_id: str, **kwargs) -> Dataset:
    """Returns a Dataset object for the existing dataset ID.
    Args:
        dataset_id (str): The dataset ID for the dataset to interact with.
    Returns:
        Dataset: A Dataset object for the existing dataset ID.
    """
    request = service_pb2.GetDatasetRequest(user_app_id=self.user_app_id, dataset_id=dataset_id)
    response = self._grpc_request(self.STUB.GetDataset, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    dict_response = MessageToDict(response)
    kwargs = self.convert_keys_to_snake_case(dict_response[list(dict_response.keys())[1]],
                                             list(dict_response.keys())[1])

    return Dataset(**kwargs)

  def model(self, model_id: str, **kwargs) -> Model:
    """Returns a Model object for the existing model ID.
    Args:
        model_id (str): The model ID for the model to interact with.
    Returns:
        Model: A Model object for the existing model ID.
    """
    request = service_pb2.GetModelRequest(user_app_id=self.user_app_id, model_id=model_id)
    response = self._grpc_request(self.STUB.GetModel, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    dict_response = MessageToDict(response)
    kwargs = self.convert_keys_to_snake_case(dict_response['model'], 'model')
    return Model(**kwargs)

  def workflow(self, workflow_id: str, **kwargs) -> Workflow:
    """Returns a workflow object for the existing workflow ID.
    Args:
        workflow_id (str): The workflow ID for the workflow to interact with.
    Returns:
        Workflow: A Workflow object for the existing workflow ID.
    """
    request = service_pb2.GetWorkflowRequest(user_app_id=self.user_app_id, workflow_id=workflow_id)
    response = self._grpc_request(self.STUB.GetWorkflow, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    dict_response = MessageToDict(response)
    kwargs = self.convert_keys_to_snake_case(dict_response[list(dict_response.keys())[1]],
                                             list(dict_response.keys())[1])

    return Workflow(**kwargs)

  def delete_dataset(self, dataset_id: str) -> None:
    """Deletes an dataset for the user.
    Args:
        dataset_id (str): The dataset ID for the app to delete.
    """
    request = service_pb2.DeleteDatasetsRequest(
        user_app_id=self.user_app_id, dataset_ids=[dataset_id])
    response = self._grpc_request(self.STUB.DeleteDatasets, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nDataset Deleted\n%s", response.status)

  def delete_model(self, model_id: str) -> None:
    """Deletes an model for the user.
    Args:
        model_id (str): The model ID for the app to delete.
    """
    request = service_pb2.DeleteModelsRequest(user_app_id=self.user_app_id, ids=[model_id])
    response = self._grpc_request(self.STUB.DeleteModels, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nModel Deleted\n%s", response.status)

  def delete_workflow(self, workflow_id: str) -> None:
    """Deletes an workflow for the user.
    Args:
        workflow_id (str): The workflow ID for the app to delete.
    """
    request = service_pb2.DeleteWorkflowsRequest(user_app_id=self.user_app_id, ids=[workflow_id])
    response = self._grpc_request(self.STUB.DeleteWorkflows, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nWorkflow Deleted\n%s", response.status)

  def __getattr__(self, name):
    return getattr(self.app_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.app_info, param)}" for param in init_params
        if hasattr(self.app_info, param)
    ]
    return f"Clarifai App Details: \n{', '.join(attribute_strings)}\n"
