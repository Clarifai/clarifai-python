from clarifai.client.auth import create_stub
from clarifai.client.auth.stub import ClarifaiAuthHelper


class BaseAuth(object):
  """
    BaseAuth is the base class for all the classes interacting with Clarifai endpoints.

    Parameters:
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
            - user_id (str): A user ID for authentication.
            - app_id (str): An app ID for the application to interact with.
            - pat (str): A personal access token for authentication.
            - base (str): The base URL for the API endpoint. Defaults to 'https://api.clarifai.com'.
            - ui (str): The URL for the UI. Defaults to 'https://clarifai.com'.

    Attributes:
        auth_helper (ClarifaiAuthHelper): An instance of ClarifaiAuthHelper for authentication.
        STUB (Stub): The gRPC Stub object for API interaction.
        metadata (tuple): The gRPC metadata containing the personal access token.
        userDataObject (UserAppIDSet): The protobuf object representing user and app IDs.
        base (str): The base URL for the API endpoint.
    """

  def __init__(self, **kwargs):
    self.auth_helper = ClarifaiAuthHelper(**kwargs)
    self.STUB = create_stub(self.auth_helper)
    self.metadata = self.auth_helper.metadata
    self.userDataObject = self.auth_helper.get_user_app_id_proto()
    self.base = self.auth_helper.base
