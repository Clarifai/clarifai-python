from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401
from clarifai.client.base import BaseClient


class Model(BaseClient):
  """
  Model is a class that provides access to Clarifai API endpoints related to Model information.
  Inherits from BaseClient for authentication purposes.
  """

  def __init__(self, model_id: str, **kwargs):
    """Initializes an Model object.
    Args:
        model_id (str): The Model ID to interact with.
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
    """
    self.kwargs = {**kwargs, 'id': model_id}
    self.model_info = resources_pb2.Model(**self.kwargs)
    super().__init__(user_id=self.user_id, app_id=self.app_id)

  def __getattr__(self, name):
    return getattr(self.model_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.model_info, param)}" for param in init_params
        if hasattr(self.model_info, param)
    ]
    return f"Model Details: \n{', '.join(attribute_strings)}\n"
