import os
import re
import time
import json
import logging
import unittest
import responses
from clarifai.rest import ApiClient

urls = [
  "https://samples.clarifai.com/metro-north.jpg",
  "https://samples.clarifai.com/wedding.jpg",
  "https://samples.clarifai.com/facebook.png",
  "https://samples.clarifai.com/dog.tiff",
  "https://samples.clarifai.com/penguin.bmp",
]


class TestApiClient(unittest.TestCase):
  """ the unit test for the ApiClient class

      This test is only to test the RESTful api endpoint calls
  """

  _multiprocess_can_split_ = True

  @classmethod
  def setUpClass(cls):
    cls.api = ApiClient(log_level=logging.WARN)

  def test_get_token(self):
    # Make sure by default the client gets a token so it can handle requests.
    resp = self.api.get_token()
    if resp != {}:
      self.assertIn('access_token', resp)
      self.assertIn('scope', resp)
      self.assertIn('expires_in', resp)

  def test_set_token(self):
    token_str = self.api.token

    if token_str:
      self.api.set_token(token_str)

  def test_api_init(self):
    api = ApiClient()

    if os.environ.get('CLARIFAI_APP_ID') and os.environ.get('CLARIFAI_APP_SECRET'):
      app_id = os.environ['CLARIFAI_APP_ID']
      app_secret = os.environ['CLARIFAI_APP_SECRET']

      api = ApiClient(app_id=app_id, app_secret=app_secret)

      if os.environ.get('CLARIFAI_API_BASE'):
        api_base = os.environ['CLARIFAI_API_BASE']
        api = ApiClient(app_id=app_id, app_secret=app_secret, base_url=api_base)

    if os.environ.get('CLARIFAI_API_KEY'):
      key = os.environ['CLARIFAI_API_KEY']

      api = ApiClient(api_key=key)

      if os.environ.get('CLARIFAI_API_BASE'):
        api_base = os.environ['CLARIFAI_API_BASE']
        api = ApiClient(api_key=key, base_url=api_base)

  def test_delete_token(self):
    self.api.delete_token()
    self.assertEqual(self.api.token, None)

    ret = self.api.get_token()
    if ret != {}:
      self.assertNotEqual(self.api.token, None)
    else:
      self.assertEqual(self.api.token, None)

  def test_add_inputs(self):
    pass

  def test_search_inputs(self):
    pass

  def test_get_input(self):
    pass

  def test_get_inputs(self):
    pass

  def test_get_inputs_status(self):
    pass

  def test_delete_input(self):
    pass

  def test_delete_inputs(self):
    pass

  def test_delete_all_inputs(self):
    # res = self.api.delete_all_inputs()
    # time.sleep(5)
    # FIXME(robert): we need a separate app to test this
    pass

  def test_update_input(self):
    pass

  def test_update_inputs(self):
    pass

  def test_get_concept(self):
    pass

  def test_get_conceps(self):
    pass

  def test_add_conceps(self):
    pass

  def test_search_conceps(self):
    pass

  def test_get_models(self):
    pass

  def test_get_model(self):
    pass

  def test_get_model_output_info(self):
    pass

  def test_get_model_versions(self):
    pass

  def test_get_model_version(self):
    pass

  def test_delete_model_version(self):
    pass

  def test_delete_model(self):
    pass

  def test_delete_all_model(self):
    # res = self.api.delete_all_models()
    # time.sleep(5)
    # FIXME(robert): we need a separate app to test this
    pass

  def test_get_model_inputs(self):
    pass

  def test_search_models(self):
    pass

  def test_create_model(self):
    pass

  def test_update_model(self):
    pass

  def test_create_model_version(self):
    pass

  def test_predict_model(self):
    pass


class TestApiExceptions(unittest.TestCase):
  """ test api errors """

  @classmethod
  def setUpClass(cls):
    cls.api = ApiClient()

  def test_bad_gateway_error(self):
    """ test bad gateway error
    """

    mocked_obj = {
      u'status':
        {
          u'code': 10020,
          u'description': u'Bad gateway'
        }
    }
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
        rsps.add(responses.GET, url_re,
                 body=mocked_json, status=status_code,
                 content_type='application/json')
        rsps.add(responses.GET, url_re,
                 body=mocked_json, status=status_code,
                 content_type='application/json')
        rsps.add(responses.GET, url_re,
                 body=mocked_good_json, status=200,
                 content_type='application/json')

        res = self.api.get_concepts(page=1, per_page=5)

        self.assertDictContainsSubset({'status': {'code': 10000, 'description': 'Ok'}}, res)
        self.assertEqual(len(res['concepts']), 5)

  def test_429_throttling(self):
    """ test 429 thrttling error
    """

    mocked_obj = {u'status':
                    {u'code': 11005,
                     u'description': u'Too Many Requests',
                     u'details': u'Making too many requests, error_details: exceeded limit '
                                 u'of 10 requests per second'
                     }
                  }
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

    with responses.RequestsMock() as rsps:
      url_re = re.compile(r'.*clarifai.com/v2/concepts.*')
      rsps.add(responses.GET, url_re,
               body=mocked_json, status=429,
               content_type='application/json')
      rsps.add(responses.GET, url_re,
               body=mocked_good_json, status=200,
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
      rsps.add(responses.GET, url_re,
               body=mocked_json, status=429,
               content_type='application/json')
      rsps.add(responses.GET, url_re,
               body=mocked_json, status=429,
               content_type='application/json')
      rsps.add(responses.GET, url_re,
               body=mocked_good_json, status=200,
               content_type='application/json')

      ts_start = time.time()
      res = self.api.get_concepts(page=1, per_page=5)
      ts_elapsed = time.time() - ts_start

      self.assertDictContainsSubset({'status': {'code': 10000, 'description': 'Ok'}}, res)
      self.assertEqual(len(res['concepts']), 5)
      self.assertGreaterEqual(ts_elapsed, 3.000)

  def test_token_expire(self):
    """ test token expiration with a mocked response
    """

    # with key authentication, there is no token and it will never expire
    ret = self.api.get_token()
    if ret == {}:
      return

    mocked_good_token_json = json.dumps(self.api.get_token())

    mocked_obj = {
      u'status':
        {
          u'code': 11001,
          u'description': u'Invalid authentication token',
          u'details': u'expired token'
        }
    }

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

    with responses.RequestsMock() as rsps:
      url_re = re.compile(r'.*clarifai.com/v2/concepts.*')
      url_re2 = re.compile(r'.*clarifai.com/v2/token.*')
      rsps.add(responses.GET, url_re,
               body=mocked_json, status=401,
               content_type='application/json')
      rsps.add(responses.GET, url_re,
               body=mocked_good_json, status=200,
               content_type='application/json')
      rsps.add(responses.POST, url_re2,
               body=mocked_good_token_json, status=200,
               content_type='application/json')

      res = self.api.get_concepts(page=1, per_page=5)

      self.assertDictContainsSubset({'status': {'code': 10000, 'description': 'Ok'}}, res)
      self.assertEqual(len(res['concepts']), 5)


if __name__ == '__main__':
  unittest.main()
