from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401

from clarifai.client.base import BaseClient


class App(BaseClient):
  """
  App is a class that provides access to Clarifai API endpoints related to App information.
  Inherits from BaseClient for authentication purposes.
  """

  def __init__(self, app_id: str, **kwargs):
    """Initializes an App object.
    Args:
        app_id (str): The App ID for the App to interact with.
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
    """
    self.kwargs = {**kwargs, 'id': app_id}
    self.app_info = resources_pb2.App(**self.kwargs)
    super().__init__(app_id=self.id)

  def __getattr__(self, name):
    return getattr(self.app_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.app_info, param)}" for param in init_params
        if hasattr(self.app_info, param)
    ]
    return f"Clarifai App Details: \n{', '.join(attribute_strings)}\n"
