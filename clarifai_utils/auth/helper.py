import os
import urllib.request
from typing import Any

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2_grpc

DEFAULT_BASE = 'api.clarifai.com'
DEFAULT_UI = 'clarifai.com'

# Map from base domain to True / False for whether the base has https or http.
# This is filled in get_stub() if it's not in there already.
base_https_cache = {}
ui_https_cache = {}


def https_cache(cache, url):
  # If http or https is provided, we trust that it is correct.
  if url.startswith('https://'):
    url = url.replace('https://', '')
    cache[url] = True
  elif url.startswith('http://'):
    url = url.replace('http://', '')
    cache[url] = False
  elif url not in cache:
    # We know our endpoints are https.
    if url.find('.clarifai.com') >= 0:
      cache[url] = True
    else:  # need to test it.
      try:  # make request to https endpoint.
        urllib.request.urlopen('https://%s/v2/auth/methods' % url, timeout=5)
        cache[url] = True  # cache it.
      except Exception as e:
        if str(e).find('SSL') >= 0:  # if ssl error then we know it's http.
          cache[url] = False
          # For http urls we need host:port format.
          if url.find(':') < 0:
            raise Exception("When providing an insecure url it must have both host:port format")
        else:
          raise Exception("Could not get a valid response from url: %s, is the API running there?"
                          % url) from e
  return url


class ClarifaiAuthHelper(object):

  def __init__(self,
               user_id: str,
               app_id: str,
               pat: str,
               token: str = '',
               base: str = DEFAULT_BASE,
               ui: str = DEFAULT_UI):
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
      base: a url to the API endpoint to hit. Examples include api.clarifai.com,
    https://api.clarifai.com (default), https://host:port, http://host:port, host:port (will be treated as http, not https). It's highly recommended to include the http:// or https:// otherwise we need to check the endpoint to determine if it has SSL during this __init__
      ui: a url to the UI. Examples include clarifai.com,
    https://clarifai.com (default), https://host:port, http://host:port, host:port (will be treated as http, not https). It's highly recommended to include the http:// or https:// otherwise we need to check the endpoint to determine if it has SSL during this __init__
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

    self._base = https_cache(base_https_cache, base)
    self._ui = https_cache(ui_https_cache, ui)

  @classmethod
  def from_streamlit(cls, st: Any, fallback_to_envvars: bool = True):
    """ This is a convenient method to check the query params from streamlit for the required
    parameters for authentication, and then optional fallback to checking environment variables as
    well if needed.
    Args:
      st: the streamlit package typically as: 'import streamlit as st'
      fallback_to_envvars: if True then when the sufficient query params are not present it will
    check env vars. If False then we raise the query param exception directly.
    Returns:
      auth: this class instantiated
    """
    try:
      auth = cls.from_streamlit_query_params(st.experimental_get_query_params())
    except Exception as e:
      if not fallback_to_envvars:
        st.error(e)
        st.stop()
        raise e
      else:
        st.markdown(
            "Could not find query params in url. For development purposes we will check for env vars next."
        )
        try:
          auth = ClarifaiAuthHelper.from_env()
        except Exception as e:
          st.error(e)
          st.stop()
          raise e
    return auth

  @classmethod
  def from_streamlit_query_params(cls, query_params: Any = ''):
    """ Initialize from streamlit queryparams. The following things will be looked for:
      user_id: as 'user_id' in query_params
      app_id: as 'app_id' in query_params
      one of:
        token: as 'token' in query_params
        pat: as 'pat' in query_params
      optionally:
        base: as 'base' in query_params.

    Args:
      query_params: the streamlit.experimental_get_query_params() response or an empty dict to fall
    back to using env vars.
    """
    error_description = """
Please check the following required query params are in the url:
 - 'user_id': the user ID accessing the module.
 - 'app_id': the app the module is being accessed from.
 - 'token' or 'pat': to authenticate the calling user with a session token or personal access token.

Additionally, these optional params are supported:
 - 'base': the base domain for the API such as https://api.clarifai.com
 - 'ui': the overall UI domain for redirects such as https://clarifai.com
"""

    if query_params == '':  # empty response from streamlit
      query_params = {}
    if 'user_id' not in query_params:
      raise Exception("You need to set 'user_id' in the query params of the url." +
                      error_description)
    user_id = query_params['user_id'][0]
    if 'app_id' not in query_params:
      raise Exception("You need to set 'app_id' in the query params of the url." +
                      error_description)
    app_id = query_params['app_id'][0]

    token = ''
    pat = ''
    if 'token' in query_params:
      token = query_params['token'][0]
    if 'pat' in query_params:
      pat = query_params['pat'][0]
    for k in ['user_id', 'app_id', 'token', 'pat']:
      if k in query_params and len(query_params[k]) != 1:
        err_str = "There should only be 1 query param value for key '%s'" % k
        raise Exception(err_str + error_description)
    if token == '' and pat == '':
      raise Exception("You must provide one of 'token' or 'pat' in the query params." +
                      error_description)
    if 'base' in query_params:
      base = query_params['base'][0]
    else:
      base = DEFAULT_BASE
    if 'ui' in query_params:
      ui = query_params['ui'][0]
    else:
      ui = DEFAULT_BASE

    return cls(user_id, app_id, pat, token, base, ui)

  @classmethod
  def from_env(cls):
    """ Will look for the following env vars:
      user_id: CLARIFAI_USER_ID env var.
      app_id: CLARIFAI_APP_ID env var.
      one of:
        token: CLARIFAI_SESSION_TOKEN env var.
        pat: CLARIFAI_PAT env var.
      base: CLARIFAI_API_BASE env var.
    """
    error_description = """
Please check the following required vars are in your env:
 - 'CLARIFAI_USER_ID': the user ID accessing the module.
 - 'CLARIFAI_APP_ID': the app the module is being accessed from.
 - 'CLARIFAI_SESSION_TOKEN' or 'CLARIFAI_PAT': to authenticate the calling user with a session token or personal access token.

Additionally, these optional params are supported:
 - 'CLARIFAI_API_BASE': the base domain for the API such as https://api.clarifai.com
 - 'CLARIFAI_UI': the overall UI domain for redirects such as https://clarifai.com
"""
    if os.environ.get('CLARIFAI_USER_ID', '') == '':
      raise Exception("You need to set the 'CLARIFAI_USER_ID' env var." + error_description)
    else:
      user_id = os.environ['CLARIFAI_USER_ID']
    if os.environ.get('CLARIFAI_APP_ID', '') == '':
      raise Exception("You need to set the 'CLARIFAI_APP_ID' env var." + error_description)
    else:
      app_id = os.environ['CLARIFAI_APP_ID']
    token = ''
    pat = ''
    if os.environ.get('CLARIFAI_SESSION_TOKEN', '') != '':
      token = os.environ['CLARIFAI_SESSION_TOKEN']
    if os.environ.get('CLARIFAI_PAT', '') != '':
      pat = os.environ['CLARIFAI_PAT']
    if token == '' and pat == '':
      raise Exception(
          "You must provide one of 'CLARIFAI_SESSION_TOKEN' or 'CLARIFAI_PAT' in your env variables."
          + error_description)
    base = os.environ.get('CLARIFAI_API_BASE', DEFAULT_BASE)
    ui = os.environ.get('CLARIFAI_UI', DEFAULT_UI)
    return cls(user_id, app_id, pat, token, base, ui)

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

  def get_stub(self):
    """
    Get the API gRPC stub using the right channel based on the API endpoint base.

    Returns:
      stub: The service_pb2_grpc.V2Stub stub for the API.
    """
    if self._base not in base_https_cache:
      raise Exception("Cannot determine if base %s is https" % self._base)

    https = base_https_cache[self._base]
    if https:
      channel = ClarifaiChannel.get_grpc_channel(base=self._base)
    else:
      if self._base.find(':') >= 0:
        host, port = self._base.split(":")
      else:
        host = self._base
        port = 80
      channel = ClarifaiChannel.get_insecure_grpc_channel(base=host, port=port)
    stub = service_pb2_grpc.V2Stub(channel)
    return stub

  @property
  def ui(self):
    return self._ui

  def __str__(self):
    return "ClarifaiAuthHelper:\n- base: %s\n- user_id: %s\n- app_id: %s\n" % (self._base,
                                                                               self.user_id,
                                                                               self.app_id)
