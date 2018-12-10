import json
import logging
import os
import re
import time
import unittest

import responses

from clarifai.rest import ApiClient, ApiError, ClarifaiApp

from . import sample_inputs


class TestApiClient(unittest.TestCase):
  """ the unit test for the ApiClient class

      This test is only to test the RESTful api endpoint calls
  """

  @classmethod
  def setUpClass(cls):
    cls.api = ApiClient(log_level=logging.WARN)

  def test_get_token(self):
    # Make sure by default the client gets a token so it can handle requests.
    with self.assertRaises(DeprecationWarning):
      self.api.get_token()

  def test_set_token(self):
    with self.assertRaises(DeprecationWarning):
      self.api.set_token('some-token')

  def test_api_init(self):
    api = ApiClient()

    if os.environ.get('CLARIFAI_API_KEY'):
      key = os.environ['CLARIFAI_API_KEY']

      api = ApiClient(api_key=key)

      if os.environ.get('CLARIFAI_API_BASE'):
        api_base = os.environ['CLARIFAI_API_BASE']
        api = ApiClient(api_key=key, base_url=api_base)

  def test_delete_token(self):
    with self.assertRaises(DeprecationWarning):
      self.api.delete_token()


class TestApiExceptions(unittest.TestCase):
  """ test api errors """

  @classmethod
  def setUpClass(cls):
    cls.api = ApiClient()

  def test_bad_gateway_error(self):
    """ test bad gateway error
    """

    mocked_obj = {u'status': {u'code': 10020, u'description': u'Bad gateway'}}
    mocked_json = json.dumps(mocked_obj)

    mocked_good_json = '''{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "concepts": [
    {
      "id": "lambdsfsdfo773337",
      "name": "lamborghini",
      "created_at": "2016-09-28T21:15:19Z",
      "app_id": "ib81c84d5b2341858b86da18a2bd21d2"
    },
    {
      "id": "lambo7werswdfw77",
      "name": "lamborghini",
      "created_at": "2016-09-28T21:15:19Z",
      "app_id": "ib81c84d5b2341858b86da18a2bd21d2"
    },
    {
      "id": "lambo777",
      "name": "lamborghini",
      "created_at": "2016-09-28T21:14:49Z",
      "app_id": "ib81c84d5b2341858b86da18a2bd21d2"
    },
    {
      "id": "mattid67",
      "name": "mattid67",
      "created_at": "2016-09-28T20:46:17Z",
      "app_id": "ib81c84d5b2341858b86da18a2bd21d2"
    },
    {
      "id": "ferrari",
      "name": "ferrari",
      "created_at": "2016-09-28T20:45:59Z",
      "app_id": "ib81c84d5b2341858b86da18a2bd21d2"
    }
  ]
}'''

    status_codes = [500, 502, 503]

    for status_code in status_codes:
      with responses.RequestsMock() as rsps:
        url_re = re.compile(r'.*clarifai.com/v2/concepts.*')
        rsps.add(
            responses.GET,
            url_re,
            body=mocked_json,
            status=status_code,
            content_type='application/json')
        rsps.add(
            responses.GET,
            url_re,
            body=mocked_json,
            status=status_code,
            content_type='application/json')
        rsps.add(
            responses.GET,
            url_re,
            body=mocked_good_json,
            status=200,
            content_type='application/json')

        res = self.api.get_concepts(page=1, per_page=5)

        self.assertDictContainsSubset({'status': {'code': 10000, 'description': 'Ok'}}, res)
        self.assertEqual(len(res['concepts']), 5)

  def test_429_throttling(self):
    """
    test that the client sleeps an appropriate length of time and retries on 429 throttling error
    """

    mocked_error_json = '''{
  "status": {
    "code": 11005,
    "description": "Too Many Requests",
    "details": "Making too many requests, error_details: exceeded limit of 10 requests per second"
  }
}'''

    mocked_success_response = '''{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "concepts": [
    {
      "id": "lambdsfsdfo773337",
      "name": "lamborghini",
      "created_at": "2016-09-28T21:15:19Z",
      "app_id": "ib81c84d5b2341858b86da18a2bd21d2"
    },
    {
      "id": "lambo7werswdfw77",
      "name": "lamborghini",
      "created_at": "2016-09-28T21:15:19Z",
      "app_id": "ib81c84d5b2341858b86da18a2bd21d2"
    },
    {
      "id": "lambo777",
      "name": "lamborghini",
      "created_at": "2016-09-28T21:14:49Z",
      "app_id": "ib81c84d5b2341858b86da18a2bd21d2"
    },
    {
      "id": "mattid67",
      "name": "mattid67",
      "created_at": "2016-09-28T20:46:17Z",
      "app_id": "ib81c84d5b2341858b86da18a2bd21d2"
    },
    {
      "id": "ferrari",
      "name": "ferrari",
      "created_at": "2016-09-28T20:45:59Z",
      "app_id": "ib81c84d5b2341858b86da18a2bd21d2"
    }
  ]
}'''

    with responses.RequestsMock() as rsps:
      url_re = re.compile(r'.*clarifai.com/v2/concepts.*')
      rsps.add(
          responses.GET,
          url_re,
          body=mocked_error_json,
          status=429,
          content_type='application/json')
      rsps.add(
          responses.GET,
          url_re,
          body=mocked_success_response,
          status=200,
          content_type='application/json')

      ts_start = time.time()
      res = self.api.get_concepts(page=1, per_page=5)
      ts_elapsed = time.time() - ts_start

    self.assertDictContainsSubset({'status': {'code': 10000, 'description': 'Ok'}}, res)
    self.assertEqual(len(res['concepts']), 5)
    self.assertGreaterEqual(ts_elapsed, 1.000)

    # another test, with two throttles
    with responses.RequestsMock() as rsps:
      url_re = re.compile(r'.*clarifai.com/v2/concepts.*')
      rsps.add(
          responses.GET,
          url_re,
          body=mocked_error_json,
          status=429,
          content_type='application/json')
      rsps.add(
          responses.GET,
          url_re,
          body=mocked_error_json,
          status=429,
          content_type='application/json')
      rsps.add(
          responses.GET,
          url_re,
          body=mocked_success_response,
          status=200,
          content_type='application/json')

      ts_start = time.time()
      res = self.api.get_concepts(page=1, per_page=5)
      ts_elapsed = time.time() - ts_start

      self.assertDictContainsSubset({'status': {'code': 10000, 'description': 'Ok'}}, res)
      self.assertEqual(len(res['concepts']), 5)
      self.assertGreaterEqual(ts_elapsed, 3.000)

  def test_invalid_base_url_error(self):

    app = ClarifaiApp()

    app.api.basev2 = 'https://nonexistent-base-url.clarifai.com'

    with self.assertRaises(ApiError):
      app.public_models.general_model.predict_by_url(sample_inputs.METRO_IMAGE_URL)

  def test_invalid_api_key_error(self):
    with self.assertRaises(ApiError) as ex:
      ClarifaiApp(api_key='some-invalid-api-key')

    raised_exception = ex.exception

    assert raised_exception.error_code == 11009
    assert raised_exception.error_desc == 'API key not found'
