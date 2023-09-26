import os
import uuid
from typing import Any, Dict, List

import yaml
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.client.base import BaseClient
from clarifai.client.dataset import Dataset
from clarifai.client.input import Inputs
from clarifai.client.lister import Lister
from clarifai.client.model import Model
from clarifai.client.module import Module
from clarifai.client.workflow import Workflow
from clarifai.errors import UserError
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.logging import display_workflow_tree, get_logger
from clarifai.workflows.utils import get_yaml_output_info_proto, is_same_yaml_model
from clarifai.workflows.validate import validate


class App(Lister, BaseClient):
  """App is a class that provides access to Clarifai API endpoints related to App information."""

  def __init__(self,
               url_init: str = "",
               app_id: str = "",
               base_url: str = "https://api.clarifai.com",
               **kwargs):
    """Initializes an App object.

    Args:
        url_init (str): The URL to initialize the app object.
        app_id (str): The App ID for the App to interact with.
        base_url (str): Base API url. Default "https://api.clarifai.com"
        **kwargs: Additional keyword arguments to be passed to the App.
            - name (str): The name of the app.
            - description (str): The description of the app.
    """
    if url_init != "" and app_id != "":
      raise UserError("You can only specify one of url_init or app_id.")
    if url_init != "":
      user_id, app_id, _, _, _ = ClarifaiUrlHelper.split_clarifai_url(url_init)
      kwargs = {'user_id': user_id}
    self.kwargs = {**kwargs, 'id': app_id}
    self.app_info = resources_pb2.App(**self.kwargs)
    self.logger = get_logger(logger_level="INFO", name=__name__)
    BaseClient.__init__(self, user_id=self.user_id, app_id=self.id, base=base_url)
    Lister.__init__(self)

  def list_datasets(self) -> List[Dataset]:
    """Lists all the datasets for the app.

    Returns:
        List[Dataset]: A list of Dataset objects for the datasets in the app.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> all_datasets = app.list_datasets()
    """
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

  def list_models(self, filter_by: Dict[str, Any] = {}, only_in_app: bool = True) -> List[Model]:
    """Lists all the available models for the user.

    Args:
        filter_by (dict): A dictionary of filters to apply to the list of models.
        only_in_app (bool): If True, only return models that are in the app.

    Returns:
        List[Model]: A list of Model objects for the models in the app.

    Example:
        >>> from clarifai.client.user import User
        >>> app = User(user_id="user_id").app(app_id="app_id")
        >>> all_models = app.list_models()
    """
    request_data = dict(user_app_id=self.user_app_id, per_page=self.default_page_size, **filter_by)
    all_models_info = list(
        self.list_all_pages_generator(self.STUB.ListModels, service_pb2.ListModelsRequest,
                                      request_data))

    filtered_models_info = []
    for model_info in all_models_info:
      if 'model_version' not in list(model_info.keys()):
        continue
      if only_in_app:
        if model_info['app_id'] != self.id:
          continue
      filtered_models_info.append(model_info)

    return [Model(**model_info) for model_info in filtered_models_info]

  def list_workflows(self, filter_by: Dict[str, Any] = {},
                     only_in_app: bool = True) -> List[Workflow]:
    """Lists all the available workflows for the user.

    Args:
        filter_by (dict): A dictionary of filters to apply to the list of workflows.
        only_in_app (bool): If True, only return workflows that are in the app.

    Returns:
        List[Workflow]: A list of Workflow objects for the workflows in the app.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> all_workflows = app.list_workflows()
    """
    request_data = dict(user_app_id=self.user_app_id, per_page=self.default_page_size, **filter_by)
    all_workflows_info = list(
        self.list_all_pages_generator(self.STUB.ListWorkflows, service_pb2.ListWorkflowsRequest,
                                      request_data))

    filtered_workflows_info = []
    for workflow_info in all_workflows_info:
      if only_in_app:
        if workflow_info['app_id'] != self.id:
          continue
      filtered_workflows_info.append(workflow_info)

    return [Workflow(**workflow_info) for workflow_info in all_workflows_info]

  def list_modules(self, filter_by: Dict[str, Any] = {}, only_in_app: bool = True) -> List[Module]:
    """Lists all the available modules for the user.

    Args:
        filter_by (dict): A dictionary of filters to apply to the list of modules.
        only_in_app (bool): If True, only return modules that are in the app.

    Returns:
        List[Module]: A list of Module objects for the modules in the app.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> all_modules = app.list_modules()
    """
    request_data = dict(user_app_id=self.user_app_id, per_page=self.default_page_size, **filter_by)
    all_modules_info = list(
        self.list_all_pages_generator(self.STUB.ListModules, service_pb2.ListModulesRequest,
                                      request_data))

    filtered_modules_info = []
    for module_info in all_modules_info:
      if only_in_app:
        if module_info['app_id'] != self.id:
          continue
      filtered_modules_info.append(module_info)

    return [Module(**module_info) for module_info in filtered_modules_info]

  def list_installed_module_versions(self, filter_by: Dict[str, Any] = {}) -> List[Module]:
    """Lists all installed module versions in the app.

    Args:
        filter_by (dict): A dictionary of filters to apply to the list of installed module versions.

    Returns:
        List[Module]: A list of Module objects for the installed module versions in the app.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> all_installed_module_versions = app.list_installed_module_versions()
    """
    request_data = dict(user_app_id=self.user_app_id, per_page=self.default_page_size, **filter_by)
    all_imv_infos = list(
        self.list_all_pages_generator(self.STUB.ListInstalledModuleVersions,
                                      service_pb2.ListInstalledModuleVersionsRequest,
                                      request_data))
    for imv_info in all_imv_infos:
      del imv_info['deploy_url']
      del imv_info['installed_module_version_id']  # TODO: remove this after the backend fix

    return [
        Module(module_id=imv_info['module_version']['module_id'], **imv_info)
        for imv_info in all_imv_infos
    ]

  def list_concepts(self):
    """Lists all the concepts for the app."""
    pass  # TODO

  def create_dataset(self, dataset_id: str, **kwargs) -> Dataset:
    """Creates a dataset for the app.

    Args:
        dataset_id (str): The dataset ID for the dataset to create.
        **kwargs: Additional keyword arguments to be passed to the Dataset.

    Returns:
        Dataset: A Dataset object for the specified dataset ID.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> dataset = app.create_dataset(dataset_id="dataset_id")
    """
    request = service_pb2.PostDatasetsRequest(
        user_app_id=self.user_app_id, datasets=[resources_pb2.Dataset(id=dataset_id, **kwargs)])
    response = self._grpc_request(self.STUB.PostDatasets, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nDataset created\n%s", response.status)
    kwargs.update({'app_id': self.id, 'user_id': self.user_id})

    return Dataset(dataset_id=dataset_id, **kwargs)

  def create_model(self, model_id: str, **kwargs) -> Model:
    """Creates a model for the app.

    Args:
        model_id (str): The model ID for the model to create.
        **kwargs: Additional keyword arguments to be passed to the Model.

    Returns:
        Model: A Model object for the specified model ID.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> model = app.create_model(model_id="model_id")
    """
    request = service_pb2.PostModelsRequest(
        user_app_id=self.user_app_id, models=[resources_pb2.Model(id=model_id, **kwargs)])
    response = self._grpc_request(self.STUB.PostModels, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nModel created\n%s", response.status)
    kwargs.update({'app_id': self.id, 'user_id': self.user_id})

    return Model(model_id=model_id, **kwargs)

  def create_workflow(self,
                      config_filepath: str,
                      generate_new_id: bool = False,
                      display: bool = True) -> Workflow:
    """Creates a workflow for the app.

    Args:
        config_filepath (str): The path to the yaml workflow config file.
        generate_new_id (bool): If True, generate a new workflow ID.
        display (bool): If True, display the workflow nodes tree.

    Returns:
        Workflow: A Workflow object for the specified workflow config.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> workflow = app.create_workflow(config_filepath="config.yml")
    """
    if not os.path.exists(config_filepath):
      raise UserError(f"Workflow config file not found at {config_filepath}")

    with open(config_filepath, 'r') as file:
      data = yaml.safe_load(file)

    data = validate(data)
    workflow = data['workflow']

    # Get all model objects from the workflow nodes.
    all_models = []
    for node in workflow['nodes']:
      output_info = get_yaml_output_info_proto(node['model'].get('output_info', None))
      try:
        model = self.model(
            node['model']['model_id'],
            node['model'].get('model_version_id', ""),
            user_id=node['model'].get('user_id', ""),
            app_id=node['model'].get('app_id', ""))
      except Exception as e:
        if "Model does not exist" in str(e):
          model = self.create_model(
              **{k: v
                 for k, v in node['model'].items() if k != 'output_info'})
          model_version = model.create_model_version(output_info=output_info)
          all_models.append(model_version.model_info)
          continue

      # If the model version ID is specified, or if the yaml model is the same as the one in the api
      if node["model"].get("model_version_id", "") or is_same_yaml_model(
          model.model_info, node["model"]):
        all_models.append(model.model_info)
      else:  # Create a new model version
        model = model.create_model_version(output_info=output_info)
        all_models.append(model.model_info)

    # Convert nodes to resources_pb2.WorkflowNodes.
    nodes = []
    for i, yml_node in enumerate(workflow['nodes']):
      node = resources_pb2.WorkflowNode(
          id=yml_node['id'],
          model=all_models[i],
      )
      # Add node inputs if they exist, i.e. if these nodes do not connect directly to the input.
      if yml_node.get("node_inputs"):
        for ni in yml_node.get("node_inputs"):
          node.node_inputs.append(resources_pb2.NodeInput(node_id=ni['node_id']))
      nodes.append(node)

    workflow_id = workflow['id']
    if generate_new_id:
      workflow_id = str(uuid.uuid4())

    # Create the workflow.
    request = service_pb2.PostWorkflowsRequest(
        user_app_id=self.user_app_id,
        workflows=[resources_pb2.Workflow(id=workflow_id, nodes=nodes)])

    response = self._grpc_request(self.STUB.PostWorkflows, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nWorkflow created\n%s", response.status)

    dict_response = MessageToDict(response, preserving_proto_field_name=True)
    # Display the workflow nodes tree.
    if display:
      display_workflow_tree(dict_response["workflows"][0]["nodes"])
    kwargs = self.process_response_keys(dict_response[list(dict_response.keys())[1]][0],
                                        "workflow")

    return Workflow(**kwargs)

  def create_module(self, module_id: str, description: str, **kwargs) -> Module:
    """Creates a module for the app.

    Args:
        module_id (str): The module ID for the module to create.
        description (str): The description of the module to create.
        **kwargs: Additional keyword arguments to be passed to the module.

    Returns:
        Module: A Module object for the specified module ID.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> module = app.create_module(module_id="module_id")
    """
    request = service_pb2.PostModulesRequest(
        user_app_id=self.user_app_id,
        modules=[resources_pb2.Module(id=module_id, description=description, **kwargs)])
    response = self._grpc_request(self.STUB.PostModules, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nModule created\n%s", response.status)
    kwargs.update({'app_id': self.id, 'user_id': self.user_id})

    return Module(module_id=module_id, **kwargs)

  def dataset(self, dataset_id: str, **kwargs) -> Dataset:
    """Returns a Dataset object for the existing dataset ID.

    Args:
        dataset_id (str): The dataset ID for the dataset to interact with.

    Returns:
        Dataset: A Dataset object for the existing dataset ID.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> dataset = app.dataset(dataset_id="dataset_id")
    """
    request = service_pb2.GetDatasetRequest(user_app_id=self.user_app_id, dataset_id=dataset_id)
    response = self._grpc_request(self.STUB.GetDataset, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    dict_response = MessageToDict(response, preserving_proto_field_name=True)
    kwargs = self.process_response_keys(dict_response[list(dict_response.keys())[1]],
                                        list(dict_response.keys())[1])

    return Dataset(**kwargs)

  def model(self, model_id: str, model_version_id: str = "", **kwargs) -> Model:
    """Returns a Model object for the existing model ID.

    Args:
        model_id (str): The model ID for the model to interact with.
        model_version_id (str): The model version ID for the model version to interact with.

    Returns:
        Model: A Model object for the existing model ID.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> model_v1 = app.model(model_id="model_id", model_version_id="model_version_id")
    """
    # Change user_app_id based on whether user_id or app_id is specified.
    if kwargs.get("user_id") or kwargs.get("app_id"):
      request = service_pb2.GetModelRequest(
          user_app_id=self.auth_helper.get_user_app_id_proto(
              kwargs.get("user_id"), kwargs.get("app_id")),
          model_id=model_id,
          version_id=model_version_id)
    else:
      request = service_pb2.GetModelRequest(
          user_app_id=self.user_app_id, model_id=model_id, version_id=model_version_id)
    response = self._grpc_request(self.STUB.GetModel, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    dict_response = MessageToDict(response, preserving_proto_field_name=True)
    kwargs = self.process_response_keys(dict_response['model'], 'model')
    return Model(**kwargs)

  def workflow(self, workflow_id: str, **kwargs) -> Workflow:
    """Returns a workflow object for the existing workflow ID.

    Args:
        workflow_id (str): The workflow ID for the workflow to interact with.

    Returns:
        Workflow: A Workflow object for the existing workflow ID.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> workflow = app.workflow(workflow_id="workflow_id")
    """
    request = service_pb2.GetWorkflowRequest(user_app_id=self.user_app_id, workflow_id=workflow_id)
    response = self._grpc_request(self.STUB.GetWorkflow, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    dict_response = MessageToDict(response, preserving_proto_field_name=True)
    kwargs = self.process_response_keys(dict_response[list(dict_response.keys())[1]],
                                        list(dict_response.keys())[1])

    return Workflow(**kwargs)

  def module(self, module_id: str, module_version_id: str = "", **kwargs) -> Module:
    """Returns a Module object for the existing module ID.

    Args:
        module_id (str): The module ID for the module to interact with.
        module_version_id (str): The module version ID for the module version to interact with.

    Returns:
        Module: A Module object for the existing module ID.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> module = app.module(module_id="module_id", module_version_id="module_version_id")
    """
    request = service_pb2.GetModuleRequest(
        user_app_id=self.user_app_id, module_id=module_id, version_id=module_version_id)
    response = self._grpc_request(self.STUB.GetModule, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    dict_response = MessageToDict(response, preserving_proto_field_name=True)
    kwargs = self.process_response_keys(dict_response['module'], 'module')

    return Module(**kwargs)

  def inputs(self,):
    """Returns an Input object.

    Returns:
        Inputs: An input object.
    """
    return Inputs(self.user_id, self.id)

  def delete_dataset(self, dataset_id: str) -> None:
    """Deletes an dataset for the user.

    Args:
        dataset_id (str): The dataset ID for the app to delete.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> app.delete_dataset(dataset_id="dataset_id")
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

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> app.delete_model(model_id="model_id")
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

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> app.delete_workflow(workflow_id="workflow_id")
    """
    request = service_pb2.DeleteWorkflowsRequest(user_app_id=self.user_app_id, ids=[workflow_id])
    response = self._grpc_request(self.STUB.DeleteWorkflows, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nWorkflow Deleted\n%s", response.status)

  def delete_module(self, module_id: str) -> None:
    """Deletes an module for the user.

    Args:
        module_id (str): The module ID for the app to delete.

    Example:
        >>> from clarifai.client.app import App
        >>> app = App(app_id="app_id", user_id="user_id")
        >>> app.delete_module(module_id="module_id")
    """
    request = service_pb2.DeleteModulesRequest(user_app_id=self.user_app_id, ids=[module_id])
    response = self._grpc_request(self.STUB.DeleteModules, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nModule Deleted\n%s", response.status)

  def __getattr__(self, name):
    return getattr(self.app_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.app_info, param)}" for param in init_params
        if hasattr(self.app_info, param)
    ]
    return f"Clarifai App Details: \n{', '.join(attribute_strings)}\n"
