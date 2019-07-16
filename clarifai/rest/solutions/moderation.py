"""
Warning: This part of the client is in beta and its public interface may still change.
"""

import logging

import requests

from clarifai.rest.http_client import HttpClient

logger = logging.getLogger('clarifai')

RETRIES = 2  # if connections fail retry a couple times.
CONNECTIONS = 20  # number of connections to maintain in pool.


class ModerationSolution(object):

  def __init__(self, api_key, base_url='https://api.clarifai-moderation.com/v2'):
    # type: (str, str) -> None

    self.api_key = api_key
    self.base_url = base_url

    session = self._make_requests_session()
    self.http_client = HttpClient(session, api_key)  # type: HttpClient

  def _make_requests_session(self):  # type: () -> requests.Session
    http_adapter = requests.adapters.HTTPAdapter(
        max_retries=RETRIES, pool_connections=CONNECTIONS, pool_maxsize=CONNECTIONS)

    session = requests.Session()
    session.mount('http://', http_adapter)
    session.mount('https://', http_adapter)
    return session

  def predict_model(self, model_id, url):  # type: (str, str) -> dict
    endpoint_url = '%s/models/%s/outputs' % (self.base_url, model_id)

    payload = {'inputs': [{'data': {'image': {'url': url}}}]}

    return self.http_client.execute_request('POST', payload, endpoint_url)

  def get_moderation_status(self, input_id):  # type: (str) -> dict
    endpoint_url = '%s/inputs/%s/outputs' % (self.base_url, input_id)

    return self.http_client.execute_request('GET', None, endpoint_url)
