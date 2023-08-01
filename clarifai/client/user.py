from typing import Any, Dict, List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401

from clarifai.client.app import App
from clarifai.client.base_client import BaseClient
from clarifai.client.lister import Lister


class User(Lister, BaseClient):
  """
  User is a class that provides access to Clarifai API endpoints related to user information.
  Inherits from BaseClient for authentication purposes.
  """

  def __init__(self, user_id: str, **kwargs):
    """Initializes a User object.
    Args:
        user_id (str): The user ID for the user to interact with.
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
    """
    self.kwargs = {**kwargs, 'id': user_id}
    self.user_info = resources_pb2.User(**self.kwargs)
    BaseClient.__init__(self, user_id=self.id)
    Lister.__init__(self)

  def list_apps(self, filter_by: Dict[str, Any] = {}) -> List[App]:
    """Lists all the apps for the user.
    Args:
        filter_by (dict): A dictionary of filters to be applied to the list of apps.
    Returns:
        list of App: A list of App objects for the user.
    """
    request_data = dict(
        user_app_id=self.userDataObject, per_page=self.default_page_size, **filter_by)
    all_apps_info = list(
        self.list_all_pages_generator(self.STUB.ListApps, service_pb2.ListAppsRequest,
                                      request_data))

    return [self.app(**app_info) for app_info in all_apps_info]

  def app(self, app_id: str, **kwargs) -> App:
    """Returns an App object for the specified app ID.
    Args:
        app_id (str): The app ID for the app to interact with.
        **kwargs: Additional keyword arguments to be passed to the App.
    Returns:
        App: An App object for the specified app ID.
    """
    kwargs['user_id'] = self.id
    return App(app_id=app_id, **kwargs)

  def __getattr__(self, name):
    return getattr(self.user_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.user_info, param)}" for param in init_params
        if hasattr(self.user_info, param)
    ]
    return f"Clarifai User Details: \n{', '.join(attribute_strings)}\n"
