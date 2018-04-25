import os
import logging
import unittest
from clarifai.rest import ClarifaiApp
from clarifai.rest import ApiError, TokenError
from past.builtins import basestring


class TestAuth(unittest.TestCase):
  """
  unit test for api auth
  """

  _multiprocess_can_split_ = True

  def test_get_token(self):
    app = ClarifaiApp(log_level=logging.WARN)
    token_str = app.auth.get_token()

    if token_str is not None:
      self.assertTrue(isinstance(token_str, basestring))
      self.assertGreaterEqual(len(token_str), 10)

  def test_auth_with_invalid_key(self):
    """ instantiate with key """

    api_preset_key = 'abc'
    with self.assertRaises(ApiError):
      app = ClarifaiApp(api_key=api_preset_key, log_level=logging.WARN)

  def test_auth_with_invalid_id_secret(self):
    with self.assertRaises(TokenError):
      app = ClarifaiApp(app_id='aa', app_secret='bb', log_level=logging.WARN)


if __name__ == '__main__':
  unittest.main()
