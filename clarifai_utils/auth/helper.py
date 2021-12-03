import os
from typing import Any

from clarifai_grpc.grpc.api import resources_pb2


class ClarifaiAuthHelper(object):

  def __init__(self, user_id: str, app_id: str, pat: str, token: str = ''):
    """
    A helper to get the authorization information needed to make API calls with the grpc
    client to a specific app using a personal access token.

    There are classmethods to make this object easily from either query_params provided by streamlit or from env vars.

    Note: only one of personal access token (pat) or a session token (token) can be provided.
    Always use PATs in your code and never session tokens, those are only provided internal UI code.

    Args:
      user_id: a user id who owns the resource you want to make calls to.
      app_id: an app id for the application that owns the resource you want to interact with
      pat: a personal access token.
      token: a session token (internal use only, always use a PAT).
    """
    if pat != "" and token != "":
      raise Exception(
          "A personal access token OR a session token need to be provided, but you cannot provide both."
      )
    elif pat == "" and token == "":
      raise Exception(
          "Need 'pat' or 'token' in the query params or use one of the CLARIFAI_PAT or CLARIFAI_SESSION_TOKEN env vars"
      )

    self.user_id = user_id
    self.app_id = app_id
    self._pat = pat
    self._token = token

  @classmethod
  def from_streamlit_query_params(cls, query_params: Any = ''):
    """ Initialize from streamlit queryparams. The following things will be looked for:
      user_id: as 'user_id' in query_params
      app_id: as 'app_id' in query_params
      token: as 'token' in query_params
      pat: as 'pat' in query_params

    Args:
      query_params: the streamlit.experimental_get_query_params() response or an empty dict to fall
    back to using env vars.
    """
    if query_params == '':  # empty response from streamlit
      query_params = {}
    if 'user_id' not in query_params:
      raise Exception("Need 'user_id' in the query params")
    if 'app_id' not in query_params:
      raise Exception("Need 'app_id' in the query params")
    token = ''
    pat = ''
    if 'token' in query_params:
      token = query_params['token'][0]
    if 'pat' in query_params:
      pat = query_params['pat'][0]
    for k in ['user_id', 'app_id', 'token', 'pat']:
      if k in query_params and len(query_params[k]) != 1:
        raise Exception("There should only be 1 query param value for key '%s'" % k)
    user_id = query_params['user_id'][0]
    app_id = query_params['app_id'][0]
    return cls(user_id, app_id, pat, token)

  @classmethod
  def from_env(cls):
    """ Will look for the following env vars:
      user_id: CLARIFAI_USER_ID env var.
      app_id: CLARIFAI_APP_ID env var.
      token: CLARIFAI_SESSION_TOKEN env var.
      pat: CLARIFAI_PAT env var.
    """
    if os.environ.get('CLARIFAI_USER_ID', '') == '':
      raise Exception("Need CLARIFAI_USER_ID env var")
    else:
      user_id = os.environ['CLARIFAI_USER_ID']
    if os.environ.get('CLARIFAI_APP_ID', '') == '':
      raise Exception("Need CLARIFAI_APP_ID env var")
    else:
      app_id = os.environ['CLARIFAI_APP_ID']
    token = ''
    pat = ''
    if os.environ.get('CLARIFAI_SESSION_TOKEN', '') != '':
      token = os.environ['CLARIFAI_SESSION_TOKEN']
    if os.environ.get('CLARIFAI_PAT', '') != '':
      pat = os.environ['CLARIFAI_PAT']
    return cls(user_id, app_id, pat, token)

  def get_user_app_id_proto(self, user_id: str = None, app_id: str = None):
    """
    Get the gRPC metadata that contains either the session token or the PAT to use.

    Args:
      user_id: optional user_id to override the default
      app_id: optional app_id to override the default.

    Returns:
      metadata: the metadata need to send with all grpc API calls in the API client.
    """
    user_id = self.user_id if user_id is None else user_id
    app_id = self.app_id if app_id is None else app_id
    return resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)

  @property
  def metadata(self):
    """
    Get the gRPC metadata that contains either the session token or the PAT to use.

    Returns:
      metadata: the metadata need to send with all grpc API calls in the API client.
    """
    if self._pat != '':
      return (('authorization', 'Key %s' % self._pat),)
    elif self._token != '':
      return (('x-clarifai-session-token', self._token),)
    else:
      raise Exception("'token' or 'pat' needed to be provided in the query params or env vars.")
