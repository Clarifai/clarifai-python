from clarifai_grpc.grpc.api import resources_pb2

from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.utils.constants import DEFAULT_BASE
from clarifai.utils.logging import logger
from clarifai.utils.protobuf import dict_to_protobuf


class Runner(Lister, BaseClient):
    """Runner is a class that provides access to Clarifai API endpoints related to Runner information."""

    def __init__(
        self,
        runner_id: str = None,
        user_id: str = None,
        base_url: str = DEFAULT_BASE,
        pat: str = None,
        token: str = None,
        root_certificates_path: str = None,
        **kwargs,
    ):
        """Initializes a Runner object.

        Args:
            runner_id (str): The Runner ID for the Runner to interact with.
            user_id (str): The user ID of the user.
            base_url (str): Base API url. Default "https://api.clarifai.com"
            pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
            token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
            root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
            **kwargs: Additional keyword arguments to be passed to the runner.
        """
        self.kwargs = {**kwargs, 'id': runner_id}
        self.runner_info = resources_pb2.Runner()
        dict_to_protobuf(self.runner_info, self.kwargs)

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

    def __getattr__(self, name):
        return getattr(self.runner_info, name)

    def __str__(self):
        init_params = [param for param in self.kwargs.keys()]
        attribute_strings = [
            f"{param}={getattr(self.runner_info, param)}"
            for param in init_params
            if hasattr(self.runner_info, param)
        ]
        return f"Runner Details: \n{', '.join(attribute_strings)}\n"
