# -*- coding: utf-8 -*-
import json
import time

import requests  # noqa
from google.protobuf.json_format import MessageToDict

from clarifai.versions import CLIENT_VERSION, OS_VER, PYTHON_VERSION


class TokenError(Exception):
  pass


class ApiError(Exception):
  """ API Server error """

  def __init__(self, resource: str, params: dict, method: str,
               response: requests.Response = None) -> None:
    self.resource = resource
    self.params = params
    self.method = method
    self.response = response

    self.error_code = 'N/A'
    self.error_desc = 'N/A'
    self.error_details = 'N/A'
    response_json = 'N/A'

    if response is not None:
      response_json_dict = MessageToDict(response)

      self.error_code = response_json_dict.get('status', {}).get('code', None)
      self.error_desc = response_json_dict.get('status', {}).get('description', None)
      self.error_details = response_json_dict.get('status', {}).get('details', None)
      response_json = json.dumps(response_json_dict['status'], indent=2)

    current_ts_str = str(time.time())

    msg = """%(method)s %(resource)s FAILED(%(time_ts)s).  error_code: %(error_code)s, error_description: %(error_desc)s, error_details: %(error_details)s
 >> Python client %(client_version)s with Python %(python_version)s on %(os_version)s
 >> %(method)s %(resource)s
 >> REQUEST(%(time_ts)s) %(request)s
 >> RESPONSE(%(time_ts)s) %(response)s""" % {
        'method': method,
        'resource': resource,
        'error_code': self.error_code,
        'error_desc': self.error_desc,
        'error_details': self.error_details,
        'request': json.dumps(params, indent=2),
        'response': response_json,
        'time_ts': current_ts_str,
        'client_version': CLIENT_VERSION,
        'python_version': PYTHON_VERSION,
        'os_version': OS_VER
    }

    super(ApiError, self).__init__(msg)


class ApiClientError(Exception):
  """ API Client Error """


class UserError(Exception):
  """ User Error """


class AuthError(Exception):
  """Raised when a client has missing or invalid authentication."""


def _base_url(url: str) -> str:
  """
  Extracts the base URL from the url, which is everything before the 4th slash character.
  https://www.clarifai.com/v2/models/1/output -> https://www.clarifai.com/v2/
  """
  try:
    return url[:_find_nth(url, '/', 4) + 1]
  except Exception:
    return ''


def _find_nth(haystack: str, needle: str, n: int) -> int:
  start = haystack.find(needle)
  while start >= 0 and n > 1:
    start = haystack.find(needle, start + len(needle))
    n -= 1
  return start
