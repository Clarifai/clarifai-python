from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401

from clarifai.client.app import App
from clarifai.client.base_auth import BaseAuth


class User(BaseAuth):
  """
  User is a class that provides access to Clarifai API endpoints related to user information.
  Inherits from BaseAuth for authentication purposes.
  """

  def __init__(self, user_id: str, **kwargs):
    """
    Initializes a User object.
    Args:
        user_id (str): The user ID for the user to interact with.
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
    """
    self.kwargs = {**kwargs, 'id': user_id}
    self.user_info = resources_pb2.User(**self.kwargs)
    super().__init__(user_id=self.id)

  def list_apps(self):
    """
    Lists all the apps for the user.
    """

  def app(self, app_id: str, **kwargs):
    """
    Returns an App object for the specified app ID.
    Args:
        app_id (str): The app ID for the app to interact with.
        **kwargs: Additional keyword arguments to be passed to the App.
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
