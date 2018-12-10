import unittest

from clarifai.rest import ApiStatus


class TestStatus(unittest.TestCase):
  """
  unit test for api status
  """

  def test_api_status(self):
    # make the status
    item = {'code': 10000, 'description': 'Ok'}

    status = ApiStatus(item)
    data = status.dict()
    self.assertEqual(10000, data['status']['code'])
    self.assertEqual('Ok', data['status']['description'])
