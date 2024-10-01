from clarifai_grpc.grpc.api import resources_pb2

from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.utils.logging import get_logger


class Deployment(Lister, BaseClient):
  """Deployment is a class that provides access to Clarifai API endpoints related to Deployment information."""

  def __init__(self,
               deployment_id: str = None,
               user_id: str = None,
               base_url: str = "https://api.clarifai.com",
               pat: str = None,
               token: str = None,
               root_certificates_path: str = None,
               **kwargs):
    """Initializes a Deployment object.

    Args:
        deployment_id (str): The Deployment ID for the Deployment to interact with.
        base_url (str): Base API url. Default "https://api.clarifai.com"
        pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
        token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
        root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
        **kwargs: Additional keyword arguments to be passed to the deployment.
    """
    self.kwargs = {**kwargs, 'id': deployment_id, 'user_id': user_id}
    self.deployment_info = resources_pb2.Deployment(**self.kwargs)
    self.logger = get_logger(logger_level="INFO", name=__name__)
    BaseClient.__init__(
        self,
        user_id=user_id,
        base=base_url,
        pat=pat,
        token=token,
        root_certificates_path=root_certificates_path)
    Lister.__init__(self)

  def __getattr__(self, name):
    return getattr(self.deployment_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.deployment_info, param)}" for param in init_params
        if hasattr(self.deployment_info, param)
    ]
    return f"Deployment Details: \n{', '.join(attribute_strings)}\n"
