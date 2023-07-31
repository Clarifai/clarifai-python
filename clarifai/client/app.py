from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401

from clarifai.client.base_auth import BaseAuth


class App(BaseAuth):
  """
  App is a class that provides access to Clarifai API endpoints related to App information.
  Inherits from BaseAuth for authentication purposes.
  """

  def __init__(self, app_id: str, **kwargs):
    """
    Initializes a App object.
    Args:
        app_id (str): The App ID for the App to interact with.
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
    """
    self.kwargs = {**kwargs, 'id': app_id}
    self.app_info = resources_pb2.App(**self.kwargs)
    super().__init__(app_id=self.id)
