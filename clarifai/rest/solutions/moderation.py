"""
Warning: This part of the client is in beta and its public interface may still change.
"""

import logging

from clarifai.rest.http_client import HttpClient

logger = logging.getLogger('clarifai')


class ModerationSolution(object):

  def __init__(self, api_key, base_url='https://api.clarifai-moderation.com/v2'):
    self.api_key = api_key
    self.base_url = base_url
    self.http_client = HttpClient(api_key)

  def predict_model(self, model_id, url):
    endpoint_url = '%s/models/%s/outputs' % (self.base_url, model_id)

    payload = {'inputs': [{'data': {'image': {'url': url}}}]}

    return self.http_client.execute_request('POST', payload, endpoint_url)

  def get_moderation_status(self, input_id):
    endpoint_url = '%s/inputs/%s/outputs' % (self.base_url, input_id)

    return self.http_client.execute_request('GET', None, endpoint_url)
