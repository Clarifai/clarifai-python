import json
import typing  # noqa
import unittest.mock as mock
from typing import Union  # noqa


class MockValidator(object):

  def __init__(self, validator):
    # validator is a function that takes a single argument and returns a bool.
    self.validator = validator

  def __eq__(self, other):
    return bool(self.validator(other))


def prepare_json(json_responses):
  if isinstance(json_responses, str):
    json_responses = [json_responses]

  # The client does the "get all models" request on initialization. This is the response returned.
  initial_get_models_response = """
  {
      "app_id": "",
      "models": []
  }
  """
  return [initial_get_models_response] + json_responses


def mock_request_with_failed_response(mock_http_client, json_responses):
  # type: (mock.Mock,typing.List[str]) -> mock.Mock
  mock_http_client.return_value.execute_request.side_effect = prepare_json(json_responses)
  return mock_http_client.return_value.execute_request


def mock_request(mock_http_client, json_responses):
  # type: (mock.Mock, Union[str, typing.List[str]]) -> mock.Mock
  json_responses = prepare_json(json_responses)

  mock_execute_request = mock_http_client.return_value.execute_request
  mock_execute_request.side_effect = [json.loads(r) for r in json_responses]
  return mock_http_client.return_value.execute_request


def mock_string_should_end_with(val):  # type: (str) -> MockValidator
  return MockValidator(lambda x: x.endswith(val))


def assert_request(mock_execute_request, method, url, json_body='{}'):
  # type: (mock.Mock, str, str, str) -> None
  assert_requests(mock_execute_request, [(method, url, json_body)])


def assert_requests(mock_execute_request, requests):
  # type: (mock.Mock, typing.List[typing.Tuple[str, str, str]]) -> None
  assert mock_execute_request.mock_calls == [
      mock.call('GET',
                json.loads('{"page": 1, "per_page": 20}'),
                mock_string_should_end_with('/v2/models'))
  ] + [
      mock.call(method, json.loads(json_body), mock_string_should_end_with(url))
      for method, url, json_body in requests
  ]
