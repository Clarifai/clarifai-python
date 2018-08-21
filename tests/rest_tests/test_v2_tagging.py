import logging
import os
import unittest

from clarifai.rest import ApiError, ClarifaiApp
from clarifai.rest import Image as ClImage
from clarifai.rest import UserError

urls = [
    "https://samples.clarifai.com/metro-north.jpg",
    "https://samples.clarifai.com/wedding.jpg",
    "https://samples.clarifai.com/facebook.png",
    "https://samples.clarifai.com/dog.tiff",
    "https://samples.clarifai.com/penguin.bmp",
]

dir_path = os.path.dirname(os.path.realpath(__file__))
files = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metro-north.jpg"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "thai-market.jpg"),
]


class TestV2Tagging(unittest.TestCase):
  """ the unit test for the V2 simplified tagging api
  """

  _multiprocess_can_split_ = True

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(log_level=logging.WARN)

  def test_tag_urls(self):
    res = self.app.tag_urls(urls)
    results = res['outputs']
    self.assertEqual(len(results), len(urls))

  def test_tag_urls_with_model_id(self):
    # general model
    res = self.app.tag_urls(urls, model_id='aaa03c23b3724a16a56b629203edc62c')
    results = res['outputs']
    self.assertEqual(len(results), len(urls))

    # travel model
    res = self.app.tag_urls(urls, model_id='eee28c313d69466f836ab83287a54ed9')
    results = res['outputs']
    self.assertEqual(len(results), len(urls))

  def test_tag_non_urls(self):
    """ tag non list or non urls """

    image_url = 'https://samples.clarifai.com/toddler-flowers.jpeg'
    urls = image_url

    with self.assertRaises(UserError) as ue:
      self.app.tag_urls(urls)

  def test_tag_overloaded_urls(self):
    """ tag more than 128 urls """

    image_url = 'https://samples.clarifai.com/toddler-flowers.jpeg'
    urls = [image_url] * 129

    with self.assertRaises(UserError) as ue:
      self.app.tag_urls(urls)

    urls = [image_url] * 16
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
    res = self.app.tag_files(files)
    results = res['outputs']
    self.assertEqual(len(results), len(files))

  def test_tag_non_files(self):
    """ tag non list or non filename """

    one_file = files[0]
    fs = one_file

    with self.assertRaises(UserError):
      self.app.tag_files(fs)

  def test_tag_overloaded_files(self):
    """ tag more than 128 files """

    file_one = files[0]
    fs = [file_one] * 129

    with self.assertRaises(UserError):
      self.app.tag_files(fs)

    fs = [file_one] * 16
    self.app.tag_files(fs)

  def test_shortcut_predict_concept_function(self):
    img1 = ClImage(url=urls[0])
    res = self.app.api.predict_concepts([img1])
    self.assertEqual(10000, res['status']['code'])

  def test_shortcut_predict_colors_function(self):
    img1 = ClImage(url=urls[0])
    res = self.app.api.predict_colors([img1])
    self.assertEqual(10000, res['status']['code'])

  def test_shortcut_predict_embed_function(self):
    img1 = ClImage(url=urls[0])
    res = self.app.api.predict_embed([img1])
    self.assertEqual(10000, res['status']['code'])


if __name__ == '__main__':
  unittest.main()
