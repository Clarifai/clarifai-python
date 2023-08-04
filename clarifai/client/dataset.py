from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401
from clarifai.client.base import BaseClient


class Dataset(BaseClient):
  """
  Dataset is a class that provides access to Clarifai API endpoints related to Dataset information.
  Inherits from BaseClient for authentication purposes.
  """

  def __init__(self, dataset_id: str, **kwargs):
    """Initializes an Dataset object.
    Args:
        dataset_id (str): The Dataset ID within the App to interact with.
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
    """
    self.kwargs = {**kwargs, 'id': dataset_id}
    self.dataset_info = resources_pb2.Dataset(**self.kwargs)
    super().__init__(user_id=self.user_id, app_id=self.app_id)

  def __getattr__(self, name):
    return getattr(self.dataset_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.dataset_info, param)}" for param in init_params
        if hasattr(self.dataset_info, param)
    ]
    return f"Dataset Details: \n{', '.join(attribute_strings)}\n"
