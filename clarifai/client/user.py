from typing import Any, Dict, List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.client.app import App
from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.client.runner import Runner
from clarifai.errors import UserError
from clarifai.utils.logging import get_logger


class User(Lister, BaseClient):
  """User is a class that provides access to Clarifai API endpoints related to user information."""

  def __init__(self, user_id: str = "", base_url: str = "https://api.clarifai.com", **kwargs):
    """Initializes an User object.

    Args:
        user_id (str): The user ID for the user to interact with.
        base_url (str): Base API url. Default "https://api.clarifai.com"
        **kwargs: Additional keyword arguments to be passed to the User.
    """
    self.kwargs = {**kwargs, 'id': user_id}
    self.user_info = resources_pb2.User(**self.kwargs)
    self.logger = get_logger(logger_level="INFO", name=__name__)
    BaseClient.__init__(self, user_id=self.id, app_id="", base=base_url)
    Lister.__init__(self)

  def list_apps(self, filter_by: Dict[str, Any] = {}) -> List[App]:
    """Lists all the apps for the user.

    Args:
        filter_by (dict): A dictionary of filters to be applied to the list of apps.

    Returns:
        list of App: A list of App objects for the user.

    Example:
        >>> from clarifai.client.user import User
        >>> apps = User("user_id").list_apps()
    """
    request_data = dict(user_app_id=self.user_app_id, per_page=self.default_page_size, **filter_by)
    all_apps_info = list(
        self.list_all_pages_generator(self.STUB.ListApps, service_pb2.ListAppsRequest,
                                      request_data))

    return [App(**app_info) for app_info in all_apps_info]

  def list_runners(self, filter_by: Dict[str, Any] = {}) -> List[Runner]:
    """List all runners for the user

    Args:
        filter_by (dict): A dictionary of filters to apply to the list of runners.

    Returns:
        List[Runner]: A list of Runner objects for the runners.

    Example:
        >>> from clarifai.client.user import User
        >>> client = User(user_id="user_id")
        >>> all_runners= client.list_runners()
    """
    request_data = dict(user_app_id=self.user_app_id, per_page=self.default_page_size, **filter_by)
    all_runners_info = list(
        self.list_all_pages_generator(self.STUB.ListRunners, service_pb2.ListRunnersRequest,
                                      request_data))

    return [Runner(check_runner_exists=False, **runner_info) for runner_info in all_runners_info]

  def create_app(self, app_id: str, base_workflow: str = 'Language-Understanding',
                 **kwargs) -> App:
    """Creates an app for the user.

    Args:
        app_id (str): The app ID for the app to create.
        base_workflow (str): The base workflow to use for the app.(Examples: 'Universal', 'Empty', 'General')
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
    kwargs.update({'user_id': self.id})
    return App(app_id=app_id, **kwargs)

  def create_runner(self, runner_id: str, labels: List[str], description: str) -> Runner:
    """Create a runner

    Args:
      runner_id (str): The Id of runner to create
      labels (List[str]): Labels to match runner
      description (str): Description of Runner

    Returns:
      Runner: A runner object for the specified Runner ID

    Example:
        >>> from clarifai.client.user import User
        >>> client = User(user_id="user_id")
        >>> runner = client.create_runner(runner_id="runner_id", labels=["label to link runner"], description="laptop runner")
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

    return Runner(
        runner_id=runner_id,
        user_id=self.id,
        labels=labels,
        description=description,
        check_runner_exists=False)

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
    return App(app_id=app_id, **kwargs)

  def runner(self, runner_id: str) -> Runner:
    """Returns a Runner object if exists.

    Args:
        runner_id (str): The runner ID to interact with

    Returns:
        Runner: A Runner object for the existing runner ID.

    Example:
        >>> from clarifai.client.user import User
        >>> client = User(user_id="user_id")
        >>> runner = client.runner(runner_id="runner_id")
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

    return Runner(check_runner_exists=False, **kwargs)

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

  def __getattr__(self, name):
    return getattr(self.user_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.user_info, param)}" for param in init_params
        if hasattr(self.user_info, param)
    ]
    return f"Clarifai User Details: \n{', '.join(attribute_strings)}\n"
