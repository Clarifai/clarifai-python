import unittest

from parameterized import parameterized

from clarifai.rest.grpc.grpc_json_channel import _pick_proper_endpoint

MODEL_VERSION_RESOURCES = [
    ('https://url.com/{model_id}/outputs', 'POST'),
    ('https://url.com/{model_id}/versions/{version_id}/outputs', 'POST'),
    ('https://url.com/{user_app_id.user_id}/apps/{user_app_id.app_id}/models/{model_id}/outputs',
     'POST'),
]

REVERSED_MODEL_VERSION_RESOURCES = [
    ('https://url.com/{model_id}/versions/{version_id}/outputs', 'POST'),
    ('https://url.com/{user_app_id.user_id}/apps/{user_app_id.app_id}/models/{model_id}/outputs',
     'POST'),
    ('https://url.com/{model_id}/outputs', 'POST'),
]

NO_ARGUMENT_RESOURCES = [('https://url.com/workflows', 'GET')]


class TestPickProperEndpoint(unittest.TestCase):

  @parameterized.expand([(MODEL_VERSION_RESOURCES,), (REVERSED_MODEL_VERSION_RESOURCES,)])
  def test_should_pick_correct_url_when_model_id(self, resources):
    request_dict = {
        'model_id': 'SOME-MODEL-ID',
    }
    picked_url = _pick_proper_endpoint(resources, request_dict)[0]
    self.assertEqual('https://url.com/SOME-MODEL-ID/outputs', picked_url)

  @parameterized.expand([(MODEL_VERSION_RESOURCES,), (REVERSED_MODEL_VERSION_RESOURCES,)])
  def test_should_pick_correct_url_when_model_id_and_version_id(self, resources):
    request_dict = {
        'model_id': 'SOME-MODEL-ID',
        'version_id': 'SOME-VERSION-ID',
    }
    picked_url = _pick_proper_endpoint(resources, request_dict)[0]
    self.assertEqual('https://url.com/SOME-MODEL-ID/versions/SOME-VERSION-ID/outputs', picked_url)

  @parameterized.expand([(MODEL_VERSION_RESOURCES,), (REVERSED_MODEL_VERSION_RESOURCES,)])
  def test_should_pick_correct_url_when_user_and_model_id(self, resources):
    request_dict = {
        'user_id': 'USER-ID',
        'app_id': 'APP-ID',
        'model_id': 'SOME-MODEL-ID',
    }
    picked_url = _pick_proper_endpoint(resources, request_dict)[0]
    self.assertEqual('https://url.com/USER-ID/apps/APP-ID/models/SOME-MODEL-ID/outputs',
                     picked_url)

  def test_should_pick_the_url_when_no_argument(self):
    request_dict = {}
    picked_url = _pick_proper_endpoint(NO_ARGUMENT_RESOURCES, request_dict)[0]
    self.assertEqual('https://url.com/workflows', picked_url)
