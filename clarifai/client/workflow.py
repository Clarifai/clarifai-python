from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401
from clarifai.client.base import BaseClient


class Workflow(BaseClient):
  """
  Workflow is a class that provides access to Clarifai API endpoints related to Workflow information.
  Inherits from BaseClient for authentication purposes.
  """

  def __init__(self, workflow_id: str, **kwargs):
    """Initializes an Workflow object.
    Args:
        workflow_id (str): The Workflow ID to interact with.
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
    """
    self.kwargs = {**kwargs, 'id': workflow_id}
    self.workflow_info = resources_pb2.Workflow(**self.kwargs)
    super().__init__(user_id=self.user_id, app_id=self.app_id)

  def __getattr__(self, name):
    return getattr(self.workflow_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.workflow_info, param)}" for param in init_params
        if hasattr(self.workflow_info, param)
    ]
    return f"Workflow Details: \n{', '.join(attribute_strings)}\n"
