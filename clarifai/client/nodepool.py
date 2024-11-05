import os
from typing import Any, Dict, Generator, List

import yaml
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.client.base import BaseClient
from clarifai.client.deployment import Deployment
from clarifai.client.lister import Lister
from clarifai.errors import UserError
from clarifai.utils.logging import logger


class Nodepool(Lister, BaseClient):
  """Nodepool is a class that provides access to Clarifai API endpoints related to Nodepool information."""

  def __init__(self,
               nodepool_id: str = None,
               user_id: str = None,
               base_url: str = "https://api.clarifai.com",
               pat: str = None,
               token: str = None,
               root_certificates_path: str = None,
               **kwargs):
    """Initializes a Nodepool object.

    Args:
        nodepool_id (str): The Nodepool ID for the Nodepool to interact with.
        user_id (str): The user ID of the user.
        base_url (str): Base API url. Default "https://api.clarifai.com"
        pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
        token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
        root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
        **kwargs: Additional keyword arguments to be passed to the nodepool.
    """
    self.kwargs = {**kwargs, 'id': nodepool_id}
    self.nodepool_info = resources_pb2.Nodepool(**self.kwargs)
    self.logger = logger
    BaseClient.__init__(
        self,
        user_id=user_id,
        base=base_url,
        pat=pat,
        token=token,
        root_certificates_path=root_certificates_path)
    Lister.__init__(self)

  def list_deployments(self,
                       filter_by: Dict[str, Any] = {},
                       page_no: int = None,
                       per_page: int = None) -> Generator[Deployment, None, None]:
    """Lists all the available deployments of compute cluster.

    Args:
        filter_by (Dict[str, Any]): The filter to apply to the list of deployments.
        page_no (int): The page number to list.
        per_page (int): The number of items per page.

    Yields:
        Deployment: Deployment objects for the nodepools in the compute cluster.

    Example:
        >>> from clarifai.client.nodepool import Nodepool
        >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
        >>> all_deployments = list(nodepool.list_deployments())

    Note:
        Defaults to 16 per page if page_no is specified and per_page is not specified.
        If both page_no and per_page are None, then lists all the resources.
    """
    request_data = dict(user_app_id=self.user_app_id, nodepool_id=self.id, **filter_by)
    all_deployments_info = self.list_pages_generator(
        self.STUB.ListDeployments,
        service_pb2.ListDeploymentsRequest,
        request_data,
        per_page=per_page,
        page_no=page_no)

    for deployment_info in all_deployments_info:
      yield Deployment.from_auth_helper(auth=self.auth_helper, **deployment_info)

  def _process_deployment_config(self, config_filepath: str) -> Dict[str, Any]:
    with open(config_filepath, "r") as file:
      deployment_config = yaml.safe_load(file)

    assert "deployment" in deployment_config, "deployment info not found in the config file"
    deployment = deployment_config['deployment']
    assert "autoscale_config" in deployment, "autoscale_config not found in the config file"
    assert ("worker" in deployment) and (
        ("model" in deployment["worker"]) or
        ("workflow" in deployment["worker"])), "worker info not found in the config file"
    assert "scheduling_choice" in deployment, "scheduling_choice not found in the config file"
    assert "nodepools" in deployment, "nodepools not found in the config file"
    deployment['user_id'] = self.user_app_id.user_id
    deployment['autoscale_config'] = resources_pb2.AutoscaleConfig(
        **deployment['autoscale_config'])
    deployment['nodepools'] = [
        resources_pb2.Nodepool(
            id=nodepool['id'],
            compute_cluster=resources_pb2.ComputeCluster(
                id=nodepool['compute_cluster']['id'], user_id=self.user_app_id.user_id))
        for nodepool in deployment['nodepools']
    ]
    if 'user' in deployment['worker']:
      deployment['worker']['user'] = resources_pb2.User(**deployment['worker']['user'])
    elif 'model' in deployment['worker']:
      deployment['worker']['model'] = resources_pb2.Model(**deployment['worker']['model'])
    elif 'workflow' in deployment['worker']:
      deployment['worker']['workflow'] = resources_pb2.Workflow(**deployment['worker']['workflow'])
    deployment['worker'] = resources_pb2.Worker(**deployment['worker'])
    if "visibility" in deployment:
      deployment["visibility"] = resources_pb2.Visibility(**deployment["visibility"])
    return deployment

  @staticmethod
  def get_runner_selector(user_id: str, compute_cluster_id: str,
                          nodepool_id: str) -> resources_pb2.RunnerSelector:
    """Returns a RunnerSelector object for the specified compute cluster and nodepool.

    Args:
        user_id (str): The user ID of the user.
        compute_cluster_id (str): The compute cluster ID for the compute cluster.
        nodepool_id (str): The nodepool ID for the nodepool.

    Returns:
        resources_pb2.RunnerSelector: A RunnerSelector object for the specified compute cluster and nodepool.

    Example:
        >>> from clarifai.client.nodepool import Nodepool
        >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
        >>> runner_selector = Nodepool.get_runner_selector(user_id="user_id", compute_cluster_id="compute_cluster_id", nodepool_id="nodepool_id")
    """
    compute_cluster = resources_pb2.ComputeCluster(id=compute_cluster_id, user_id=user_id)
    nodepool = resources_pb2.Nodepool(id=nodepool_id, compute_cluster=compute_cluster)
    return resources_pb2.RunnerSelector(nodepool=nodepool)

  def create_deployment(self, config_filepath: str, deployment_id: str = None) -> Deployment:
    """Creates a deployment for the nodepool.

    Args:
        config_filepath (str): The path to the deployment config file.
        deployment_id (str): New deployment ID for the deployment to create.

    Returns:
        Deployment: A Deployment object for the specified deployment ID.

    Example:
        >>> from clarifai.client.nodepool import Nodepool
        >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
        >>> deployment = nodepool.create_deployment(config_filepath="config.yml")
    """
    if not os.path.exists(config_filepath):
      raise UserError(f"Deployment config file not found at {config_filepath}")

    deployment_config = self._process_deployment_config(config_filepath)

    if 'id' in deployment_config:
      if deployment_id is None:
        deployment_id = deployment_config['id']
      deployment_config.pop('id')

    request = service_pb2.PostDeploymentsRequest(
        user_app_id=self.user_app_id,
        deployments=[resources_pb2.Deployment(id=deployment_id, **deployment_config)])
    response = self._grpc_request(self.STUB.PostDeployments, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nDeployment created\n%s", response.status)

    return Deployment.from_auth_helper(self.auth_helper, deployment_id=deployment_id)

  def deployment(self, deployment_id: str) -> Deployment:
    """Returns a Deployment object for the existing deployment ID.

    Args:
        deployment_id (str): The deployment ID for the deployment to interact with.

    Returns:
        Deployment: A Deployment object for the existing deployment ID.

    Example:
        >>> from clarifai.client.nodepool import Nodepool
        >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
        >>> deployment = nodepool.deployment(deployment_id="deployment_id")
    """
    request = service_pb2.GetDeploymentRequest(
        user_app_id=self.user_app_id, deployment_id=deployment_id)
    response = self._grpc_request(self.STUB.GetDeployment, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    dict_response = MessageToDict(response, preserving_proto_field_name=True)
    kwargs = self.process_response_keys(dict_response[list(dict_response.keys())[1]],
                                        list(dict_response.keys())[1])

    return Deployment.from_auth_helper(auth=self.auth_helper, **kwargs)

  def delete_deployments(self, deployment_ids: List[str]) -> None:
    """Deletes list of deployments for the nodepool.

    Args:
        deployment_ids (List[str]): The list of deployment IDs to delete.

    Example:
        >>> from clarifai.client.nodepool import Nodepool
        >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
        >>> nodepool.delete_deployments(deployment_ids=["deployment_id1", "deployment_id2"])
    """
    assert isinstance(deployment_ids, list), "deployment_ids param should be a list"

    request = service_pb2.DeleteDeploymentsRequest(
        user_app_id=self.user_app_id, ids=deployment_ids)
    response = self._grpc_request(self.STUB.DeleteDeployments, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nDeployments Deleted\n%s", response.status)

  def __getattr__(self, name):
    return getattr(self.nodepool_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.nodepool_info, param)}" for param in init_params
        if hasattr(self.nodepool_info, param)
    ]
    return f"Nodepool Details: \n{', '.join(attribute_strings)}\n"
