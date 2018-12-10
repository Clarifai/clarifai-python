import logging
import unittest

from clarifai.rest import ApiError, ClarifaiApp


class TestAuth(unittest.TestCase):
  """
  unit test for api auth
  """

  def test_auth_with_invalid_key(self):
    """ instantiate with key """

    api_preset_key = 'abc'
    with self.assertRaises(ApiError):
      ClarifaiApp(api_key=api_preset_key, log_level=logging.WARN)

  def test_auth_with_id_and_secret(self):
    with self.assertRaises(DeprecationWarning):
      ClarifaiApp(app_id='aa', app_secret='bb', log_level=logging.WARN)
