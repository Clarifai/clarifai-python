import os
from typing import Any, Dict, Generator, List

import yaml
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict
from google.protobuf.wrappers_pb2 import BoolValue

from clarifai.client.app import App
from clarifai.client.base import BaseClient
from clarifai.client.compute_cluster import ComputeCluster
from clarifai.client.lister import Lister
from clarifai.errors import UserError
from clarifai.utils.logging import logger


class User(Lister, BaseClient):
  """User is a class that provides access to Clarifai API endpoints related to user information."""

  def __init__(self,
               user_id: str = None,
               base_url: str = "https://api.clarifai.com",
               pat: str = None,
               token: str = None,
               root_certificates_path: str = None,
               **kwargs):
    """Initializes an User object.

    Args:
        user_id (str): The user ID for the user to interact with.
        base_url (str): Base API url. Default "https://api.clarifai.com"
        pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
        token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
        root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
        **kwargs: Additional keyword arguments to be passed to the User.
    """
    self.kwargs = {**kwargs, 'id': user_id}
    self.user_info = resources_pb2.User(**self.kwargs)
    self.logger = logger
    BaseClient.__init__(
        self,
        user_id=self.id,
        app_id="",
        base=base_url,
        pat=pat,
        token=token,
        root_certificates_path=root_certificates_path)
    Lister.__init__(self)

  def list_apps(self, filter_by: Dict[str, Any] = {}, page_no: int = None,
                per_page: int = None) -> Generator[App, None, None]:
    """Lists all the apps for the user.

    Args:
        filter_by (dict): A dictionary of filters to be applied to the list of apps.
        page_no (int): The page number to list.
        per_page (int): The number of items per page.

    Yields:
        App: App objects for the user.

    Example:
        >>> from clarifai.client.user import User
        >>> apps = list(User("user_id").list_apps())

    Note:
        Defaults to 16 per page if page_no is specified and per_page is not specified.
        If both page_no and per_page are None, then lists all the resources.
    """
    request_data = dict(user_app_id=self.user_app_id, **filter_by)
    all_apps_info = self.list_pages_generator(
        self.STUB.ListApps,
        service_pb2.ListAppsRequest,
        request_data,
        per_page=per_page,
        page_no=page_no)
    for app_info in all_apps_info:
      yield App.from_auth_helper(
          self.auth_helper,
          **app_info)  #(base_url=self.base, pat=self.pat, token=self.token, **app_info)

  def list_runners(self, filter_by: Dict[str, Any] = {}, page_no: int = None,
                   per_page: int = None) -> Generator[dict, None, None]:
    """List all runners for the user

    Args:
        filter_by (dict): A dictionary of filters to apply to the list of runners.
        page_no (int): The page number to list.
        per_page (int): The number of items per page.

    Yields:
        Dict: Dictionaries containing information about the runners.

    Example:
        >>> from clarifai.client.user import User
        >>> client = User(user_id="user_id")
        >>> all_runners= list(client.list_runners())

    Note:
        Defaults to 16 per page if page_no is specified and per_page is not specified.
        If both page_no and per_page are None, then lists all the resources.
    """
    request_data = dict(user_app_id=self.user_app_id, **filter_by)
    all_runners_info = self.list_pages_generator(
        self.STUB.ListRunners,
        service_pb2.ListRunnersRequest,
        request_data,
        per_page=per_page,
        page_no=page_no)

    for runner_info in all_runners_info:
      yield dict(auth=self.auth_helper, check_runner_exists=False, **runner_info)

  def list_compute_clusters(self, page_no: int = None,
                            per_page: int = None) -> Generator[dict, None, None]:
    """List all compute clusters for the user

    Args:
        page_no (int): The page number to list.
        per_page (int): The number of items per page.

    Yields:
        Dict: Dictionaries containing information about the compute clusters.

    Example:
        >>> from clarifai.client.user import User
        >>> client = User(user_id="user_id")
        >>> all_compute_clusters= list(client.list_compute_clusters())

    Note:
        Defaults to 16 per page if page_no is specified and per_page is not specified.
        If both page_no and per_page are None, then lists all the resources.
    """
    request_data = dict(user_app_id=self.user_app_id)
    all_compute_clusters_info = self.list_pages_generator(
        self.STUB.ListComputeClusters,
        service_pb2.ListComputeClustersRequest,
        request_data,
        per_page=per_page,
        page_no=page_no)

    for compute_cluster_info in all_compute_clusters_info:
      yield ComputeCluster.from_auth_helper(self.auth_helper, **compute_cluster_info)

  def create_app(self, app_id: str, base_workflow: str = 'Empty', **kwargs) -> App:
    """Creates an app for the user.

    Args:
        app_id (str): The app ID for the app to create.
        base_workflow (str): The base workflow to use for the app.(Examples: 'Universal', 'Language-Understanding', 'General')
        **kwargs: Additional keyword arguments to be passed to the App.

    Returns:
        App: An App object for the specified app ID.

    Example:
        >>> from clarifai.client.user import User
        >>> client = User(user_id="user_id")
        >>> app = client.create_app(app_id="app_id",base_workflow="Universal")
    """
    workflow = resources_pb2.Workflow(id=base_workflow, app_id="main", user_id="clarifai")
    request = service_pb2.PostAppsRequest(
        user_app_id=self.user_app_id,
        apps=[resources_pb2.App(id=app_id, default_workflow=workflow, **kwargs)])
    response = self._grpc_request(self.STUB.PostApps, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nApp created\n%s", response.status)
    return App.from_auth_helper(auth=self.auth_helper, app_id=app_id)

  def create_runner(self, runner_id: str, labels: List[str], description: str) -> dict:
    """Create a runner

    Args:
      runner_id (str): The Id of runner to create
      labels (List[str]): Labels to match runner
      description (str): Description of Runner

    Returns:
      Dict: A dictionary containing information about the specified Runner ID.

    Example:
        >>> from clarifai.client.user import User
        >>> client = User(user_id="user_id")
        >>> runner_info = client.create_runner(runner_id="runner_id", labels=["label to link runner"], description="laptop runner")
    """

    if not isinstance(labels, List):
      raise UserError("Labels must be a List of strings")

    request = service_pb2.PostRunnersRequest(
        user_app_id=self.user_app_id,
        runners=[resources_pb2.Runner(id=runner_id, labels=labels, description=description)])
    response = self._grpc_request(self.STUB.PostRunners, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nRunner created\n%s", response.status)

    return dict(
        auth=self.auth_helper,
        runner_id=runner_id,
        user_id=self.id,
        labels=labels,
        description=description,
        check_runner_exists=False)

  def _process_compute_cluster_config(self, config_filepath: str) -> Dict[str, Any]:
    with open(config_filepath, "r") as file:
      compute_cluster_config = yaml.safe_load(file)

    assert "compute_cluster" in compute_cluster_config, "compute cluster info not found in the config file"
    compute_cluster = compute_cluster_config['compute_cluster']
    assert "region" in compute_cluster, "region not found in the config file"
    assert "managed_by" in compute_cluster, "managed_by not found in the config file"
    assert "cluster_type" in compute_cluster, "cluster_type not found in the config file"
    compute_cluster['cloud_provider'] = resources_pb2.CloudProvider(
        **compute_cluster['cloud_provider'])
    compute_cluster['key'] = resources_pb2.Key(id=self.pat)
    if "visibility" in compute_cluster:
      compute_cluster["visibility"] = resources_pb2.Visibility(**compute_cluster["visibility"])
    return compute_cluster

  def create_compute_cluster(self, compute_cluster_id: str,
                             config_filepath: str) -> ComputeCluster:
    """Creates a compute cluster for the user.

    Args:
        compute_cluster_id (str): The compute cluster ID for the compute cluster to create.
        config_filepath (str): The path to the compute cluster config file.

    Returns:
        ComputeCluster: A Compute Cluster object for the specified compute cluster ID.

    Example:
        >>> from clarifai.client.user import User
        >>> client = User(user_id="user_id")
        >>> compute_cluster = client.create_compute_cluster(compute_cluster_id="compute_cluster_id", config_filepath="config.yml")
    """
    if not os.path.exists(config_filepath):
      raise UserError(f"Compute Cluster config file not found at {config_filepath}")

    compute_cluster_config = self._process_compute_cluster_config(config_filepath)

    if 'id' in compute_cluster_config:
      compute_cluster_id = compute_cluster_config['id']
      compute_cluster_config.pop('id')

    request = service_pb2.PostComputeClustersRequest(
        user_app_id=self.user_app_id,
        compute_clusters=[
            resources_pb2.ComputeCluster(id=compute_cluster_id, **compute_cluster_config)
        ])
    response = self._grpc_request(self.STUB.PostComputeClusters, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nCompute Cluster created\n%s", response.status)
    return ComputeCluster.from_auth_helper(
        auth=self.auth_helper, compute_cluster_id=compute_cluster_id)

  def app(self, app_id: str, **kwargs) -> App:
    """Returns an App object for the specified app ID.

    Args:
        app_id (str): The app ID for the app to interact with.
        **kwargs: Additional keyword arguments to be passed to the App.

    Returns:
        App: An App object for the specified app ID.

    Example:
        >>> from clarifai.client.user import User
        >>> app = User("user_id").app("app_id")
    """
    request = service_pb2.GetAppRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=self.id, app_id=app_id))
    response = self._grpc_request(self.STUB.GetApp, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)

    kwargs['user_id'] = self.id
    return App.from_auth_helper(auth=self.auth_helper, app_id=app_id, **kwargs)

  def runner(self, runner_id: str) -> dict:
    """Returns a Runner object if exists.

    Args:
        runner_id (str): The runner ID to interact with

    Returns:
        Dict: A dictionary containing information about the existing runner ID.

    Example:
        >>> from clarifai.client.user import User
        >>> client = User(user_id="user_id")
        >>> runner_info = client.runner(runner_id="runner_id")
    """
    request = service_pb2.GetRunnerRequest(user_app_id=self.user_app_id, runner_id=runner_id)
    response = self._grpc_request(self.STUB.GetRunner, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(
          f"""Error getting runner, are you use this is a valid runner id {runner_id} at the user_id/app_id
          {self.user_app_id.user_id}/{self.user_app_id.app_id}.
          Error: {response.status.description}""")

    dict_response = MessageToDict(response, preserving_proto_field_name=True)
    kwargs = self.process_response_keys(dict_response[list(dict_response.keys())[1]],
                                        list(dict_response.keys())[1])

    return dict(self.auth_helper, check_runner_exists=False, **kwargs)

  def compute_cluster(self, compute_cluster_id: str) -> ComputeCluster:
    """Returns an Compute Cluster object for the specified compute cluster ID.

    Args:
        compute_cluster_id (str): The compute cluster ID for the compute cluster to interact with.

    Returns:
        ComputeCluster: A Compute Cluster object for the specified compute cluster ID.

    Example:
        >>> from clarifai.client.user import User
        >>> compute_cluster = User("user_id").compute_cluster("compute_cluster_id")
    """
    request = service_pb2.GetComputeClusterRequest(
        user_app_id=self.user_app_id, compute_cluster_id=compute_cluster_id)
    response = self._grpc_request(self.STUB.GetComputeCluster, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)

    dict_response = MessageToDict(response, preserving_proto_field_name=True)
    kwargs = self.process_response_keys(dict_response[list(dict_response.keys())[1]],
                                        list(dict_response.keys())[1])

    return ComputeCluster.from_auth_helper(auth=self.auth_helper, **kwargs)

  def patch_app(self, app_id: str, action: str = 'overwrite', **kwargs) -> App:
    """Patch an app for the user.

    Args:
        app_id (str): The app ID for the app to patch.
        action (str): The action to perform on the app (overwrite/remove).
        **kwargs: Additional keyword arguments to be passed to patch the App.

    Returns:
        App: Patched App object for the specified app ID.
    """
    if "base_workflow" in kwargs:
      kwargs["default_workflow"] = resources_pb2.Workflow(
          id=kwargs.pop("base_workflow"), app_id="main", user_id="clarifai")
    if "visibility" in kwargs:
      kwargs["visibility"] = resources_pb2.Visibility(gettable=kwargs["visibility"])
    if "image_url" in kwargs:
      kwargs["image"] = resources_pb2.Image(url=kwargs.pop("image_url"))
    if "is_template" in kwargs:
      kwargs["is_template"] = BoolValue(value=kwargs["is_template"])
    request = service_pb2.PatchAppRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=self.id, app_id=app_id),
        app=resources_pb2.App(id=app_id, **kwargs),
        action=action,
        reindex=False)
    response = self._grpc_request(self.STUB.PatchApp, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nApp patched\n%s", response.status)

    return App.from_auth_helper(auth=self.auth_helper, app_id=app_id)

  def delete_app(self, app_id: str) -> None:
    """Deletes an app for the user.

    Args:
        app_id (str): The app ID for the app to delete.

    Example:
        >>> from clarifai.client.user import User
        >>> user = User("user_id").delete_app("app_id")
    """
    request = service_pb2.DeleteAppRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=self.id, app_id=app_id))
    response = self._grpc_request(self.STUB.DeleteApp, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nApp Deleted\n%s", response.status)

  def delete_runner(self, runner_id: str) -> None:
    """Deletes all spectified runner ids

    Args:
        runner_ids (str): List of runners to delete

    Example:
        >>> from clarifai.client.user import User
        >>> client = User(user_id="user_id")
        >>> client.delete_runner(runner_id="runner_id")
    """
    request = service_pb2.DeleteRunnersRequest(user_app_id=self.user_app_id, ids=[runner_id])
    response = self._grpc_request(self.STUB.DeleteRunners, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nRunner Deleted\n%s", response.status)

  def delete_compute_clusters(self, compute_cluster_ids: List[str]) -> None:
    """Deletes a list of compute clusters for the user.

    Args:
        compute_cluster_ids (List[str]): The compute cluster IDs of the user to delete.

    Example:
        >>> from clarifai.client.user import User
        >>> user = User("user_id").delete_compute_clusters(compute_cluster_ids=["compute_cluster_id1", "compute_cluster_id2"])
    """
    assert isinstance(compute_cluster_ids, list), "compute_cluster_ids param should be a list"

    request = service_pb2.DeleteComputeClustersRequest(
        user_app_id=self.user_app_id, ids=compute_cluster_ids)
    response = self._grpc_request(self.STUB.DeleteComputeClusters, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nCompute Cluster Deleted\n%s", response.status)

  def __getattr__(self, name):
    return getattr(self.user_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.user_info, param)}" for param in init_params
        if hasattr(self.user_info, param)
    ]
    return f"Clarifai User Details: \n{', '.join(attribute_strings)}\n"
