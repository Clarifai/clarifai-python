from clarifai_grpc.grpc.api import resources_pb2

from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.utils.logging import logger


class Runner(Lister, BaseClient):
  """Nodepool is a class that provides access to Clarifai API endpoints related to Nodepool information."""

  def __init__(
      self,
      runner_id: str = None,
      user_id: str = None,
      base_url: str = "https://api.clarifai.com",
      pat: str = None,
      token: str = None,
      root_certificates_path: str = None,
      **kwargs,
  ):
    """Initializes a Nodepool object.

    Args:
        nodepool_id (str): The Nodepool ID for the Nodepool to interact with.
        user_id (str): The user ID of the user.
        base_url (str): Base API url. Default "https://api.clarifai.com"
        pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
        token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
        root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
        **kwargs: Additional keyword arguments to be passed to the nodepool.
    """
    self.kwargs = {**kwargs, 'id': runner_id}
    self.nodepool_info = resources_pb2.Runner(**self.kwargs)
    self.logger = logger
    BaseClient.__init__(
        self,
        user_id=user_id,
        base=base_url,
        pat=pat,
        token=token,
        root_certificates_path=root_certificates_path,
    )
    Lister.__init__(self)
