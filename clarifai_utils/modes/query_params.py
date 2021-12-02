import os

from clarifai_grpc.grpc.api import resources_pb2


class ClarifaiAppModeQueryParams(object):

  def __init__(self, query_params=''):
    """
    Args:
      query_params: the streamlit.experimental_get_query_params() response or an empty dict to fall
    back to using env vars.
    """
    if query_params == '':  # empty response from streamlit
      query_params = {}

    if 'user_id' not in query_params:
      if os.environ.get('CLARIFAI_USER_ID', '') == '':
        raise Exception("Need 'user_id' in the query params or CLARIFAI_USER_ID")
      else:
        query_params['user_id'] = [os.environ['CLARIFAI_USER_ID']]
    if 'app_id' not in query_params:
      if os.environ.get('CLARIFAI_APP_ID', '') == '':
        raise Exception("Need 'app_id' in the query params or CLARIFAI_APP_ID")
      else:
        query_params['app_id'] = [os.environ['CLARIFAI_APP_ID']]
    if 'token' not in query_params:
      if os.environ.get('CLARIFAI_SESSION_TOKEN', '') == '':
        raise Exception("Need 'token' in the query params or CLARIFAI_SESSION_TOKEN")
      else:
        query_params['token'] = [os.environ['CLARIFAI_SESSION_TOKEN']]
    for k in ['user_id', 'app_id', 'token']:
      if len(query_params[k]) != 1:
        raise Exception("There should only be 1 query param value for key '%s'" % k)

    self.st_query_params = query_params

    self._user_app_id_proto = resources_pb2.UserAppIDSet(user_id=self.user_id, app_id=self.app_id)

  @property
  def token(self):
    return self.st_query_params['token'][0]

  @property
  def user_id(self):
    return self.st_query_params['user_id'][0]

  @property
  def app_id(self):
    return self.st_query_params['app_id'][0]

  @property
  def user_app_id_proto(self):
    return self._user_app_id_proto
