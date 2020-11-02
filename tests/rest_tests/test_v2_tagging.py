import logging
import unittest

from clarifai.rest import ApiError, ClarifaiApp
from clarifai.rest import Image as ClImage
from clarifai.rest import UserError

from . import sample_inputs

URLS = [
    sample_inputs.METRO_IMAGE_URL,
    sample_inputs.WEDDING_IMAGE_URL,
    sample_inputs.FACEBOOK_IMAGE_URL,
    sample_inputs.DOG_TIFF_IMAGE_URL,
    sample_inputs.DOG2_IMAGE_URL,
    sample_inputs.DOG2_NONEXISTENT_IMAGE_URL,
    sample_inputs.PENGUIN_BMP_IMAGE_URL,
]

FILE_PATHS = [
    sample_inputs.METRO_IMAGE_FILE_PATH,
    sample_inputs.THAI_MARKET_IMAGE_FILE_PATH,
]


# Tagging is deprecated but the tests are kept to ensure backward-compatibility.
class TestV2Tagging(unittest.TestCase):
  """ the unit test for the V2 simplified tagging api
  """

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(log_level=logging.WARN)

  def test_tag_urls(self):
    res = self.app.tag_urls(URLS)
    results = res['outputs']
    self.assertEqual(len(results), len(URLS))

  def test_tag_urls_with_model_id(self):
    # general model
    res = self.app.tag_urls(URLS, model_id='aaa03c23b3724a16a56b629203edc62c')
    results = res['outputs']
    self.assertEqual(len(results), len(URLS))

    # travel model
    res = self.app.tag_urls(URLS, model_id='eee28c313d69466f836ab83287a54ed9')
    results = res['outputs']
    self.assertEqual(len(results), len(URLS))

  def test_tag_non_urls(self):
    """ tag non list or non urls """

    with self.assertRaises(UserError) as ue:
      self.app.tag_urls(sample_inputs.TODDLER_FLOWERS_IMAGE_URL)

  def test_tag_overloaded_urls(self):
    """ tag more than 128 urls """

    urls = [sample_inputs.TODDLER_FLOWERS_IMAGE_URL] * 129

    with self.assertRaises(UserError) as ue:
      self.app.tag_urls(urls)

    urls = [sample_inputs.TODDLER_FLOWERS_IMAGE_URL] * 16
    try:
      res = self.app.tag_urls(urls)
    except ApiError as e:
      # ignore the partial errors
      if e.response.status_code == 207:
        pass
      else:
        raise e

    self.assertEqual(len(res['outputs']), len(urls))

  def test_tag_files(self):
    res = self.app.tag_files(FILE_PATHS)
    results = res['outputs']
    self.assertEqual(len(results), len(FILE_PATHS))

  def test_tag_non_files(self):
    """ tag non list or non filename """

    with self.assertRaises(UserError):
      self.app.tag_files(sample_inputs.METRO_IMAGE_FILE_PATH)

  def test_tag_overloaded_files(self):
    """ tag more than 128 files """

    fs = [sample_inputs.METRO_IMAGE_FILE_PATH] * 129

    with self.assertRaises(UserError):
      self.app.tag_files(fs)

    fs = [sample_inputs.METRO_IMAGE_FILE_PATH] * 16
    self.app.tag_files(fs)

  def test_shortcut_predict_concept_function(self):
    img1 = ClImage(url=sample_inputs.METRO_IMAGE_URL)
    res = self.app.api.predict_concepts([img1])
    self.assertEqual(10000, res['status']['code'])

  def test_shortcut_predict_colors_function(self):
    img1 = ClImage(url=sample_inputs.METRO_IMAGE_URL)
    res = self.app.api.predict_colors([img1])
    self.assertEqual(10000, res['status']['code'])

  def test_shortcut_predict_embed_function(self):
    img1 = ClImage(url=sample_inputs.METRO_IMAGE_URL)
    res = self.app.api.predict_embed([img1])
    self.assertEqual(10000, res['status']['code'])
