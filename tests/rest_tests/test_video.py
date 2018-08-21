# -*- coding: utf-8 -*-

import base64
import logging
import os
import tempfile
import unittest

from clarifai.rest import ClarifaiApp
from clarifai.rest import Video as ClVideo

video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'small.mp4')


class TestVideo(unittest.TestCase):
  """
  unit test for video related calls
  """

  _multiprocess_can_split_ = True

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(log_level=logging.WARN)
    cls.m = cls.app.models.get('general-v1.3')

  def test_create_video_obj(self):
    """ test creating Clarifai Video object from different sources """
    url = 'https://samples.clarifai.com/3o6gb3kkXfLvdKEZs4.gif'
    v1 = ClVideo(url=url)

    self.assertTrue(isinstance(v1, ClVideo))
    self.assertEqual(v1.url, url)

    v1_json = v1.dict()
    self.assertIn('video', v1_json['data'])
    self.assertIn('url', v1_json['data']['video'])

    # test url with leading or trailing spaces
    url = 'https://samples.clarifai.com/3o6gb3kkXfLvdKEZs4.gif'
    v1 = ClVideo(url=' ' + url)
    self.assertEqual(v1.url, url)
    v1 = ClVideo(url=' ' + url + '  ')
    self.assertEqual(v1.url, url)

    with open(video_path, 'rb') as f:
      video_file_bytes = f.read()

    # test video file
    v2 = ClVideo(filename=video_path)

    toddler_flowers_base64_bytes = base64.b64encode(video_file_bytes)
    v3 = ClVideo(base64=toddler_flowers_base64_bytes)

    with open(video_path, 'rb') as f:
      v4 = ClVideo(file_obj=f)

  def _verify_video_response(self, video):
    """ validate if the response is a video """
    self.assertIn('outputs', video)
    self.assertGreater(len(video['outputs']), 0)

    result0 = video['outputs'][0]
    self.assertIn('video', result0['input']['data'])
    self.assertIn('frames', result0['data'])

  def test_predict_url(self):
    """ test predict video from url """

    url = 'https://samples.clarifai.com/3o6gb3kkXfLvdKEZs4.gif'
    res = self.m.predict_by_url(url, is_video=True)
    self._verify_video_response(res)

    url = 'https://samples.clarifai.com/beer.mp4'
    res = self.m.predict_by_url(url, is_video=True)
    self._verify_video_response(res)

  def test_predict_with_raw_bytes(self):
    """ test video predict with other sources rather than url """

    url = 'https://s3.amazonaws.com/samples.clarifai.com/D7qTae7IQLKSI.gif'

    # predict by file raw bytes
    raw_bytes = self.app.api.session.get(url).content
    res = self.m.predict_by_bytes(raw_bytes, is_video=True)
    self._verify_video_response(res)

  def test_predict_with_base64_bytes(self):
    url = 'https://s3.amazonaws.com/samples.clarifai.com/D7qTae7IQLKSI.gif'

    raw_bytes = self.app.api.session.get(url).content
    base64_bytes = base64.b64encode(raw_bytes)
    res = self.m.predict_by_base64(base64_bytes, is_video=True)
    self._verify_video_response(res)

  def test_predict_with_file(self):
    url = 'https://s3.amazonaws.com/samples.clarifai.com/D7qTae7IQLKSI.gif'

    raw_bytes = self.app.api.session.get(url).content
    # predict by local filename
    f = tempfile.NamedTemporaryFile(delete=False)
    filename = f.name
    f.write(raw_bytes)
    f.close()

    res = self.m.predict_by_filename(filename, is_video=True)
    self._verify_video_response(res)

    os.unlink(filename)

  def test_video_with_apparel(self):
    """ test video with not just general models """

    m = self.app.models.get('apparel')
    url = 'https://s3.amazonaws.com/samples.clarifai.com/D7qTae7IQLKSI.gif'
    res = m.predict_by_url(url, is_video=True)
    self._verify_video_response(res)

  def test_video_with_nsfw(self):
    """ test video with not just general models """

    m = self.app.models.get('nsfw-v1.0')
    url = 'https://s3.amazonaws.com/samples.clarifai.com/D7qTae7IQLKSI.gif'
    res = m.predict_by_url(url, is_video=True)
    self._verify_video_response(res)

  def test_video_with_wedding(self):
    """ test video with not just general models """

    m = self.app.models.get('weddings-v1.0')
    url = 'https://s3.amazonaws.com/samples.clarifai.com/D7qTae7IQLKSI.gif'
    res = m.predict_by_url(url, is_video=True)
    self._verify_video_response(res)

  def test_video_with_food(self):
    """ test video with not just general models """

    m = self.app.models.get('food-items-v1.0')
    url = 'https://s3.amazonaws.com/samples.clarifai.com/D7qTae7IQLKSI.gif'
    res = m.predict_by_url(url, is_video=True)
    self._verify_video_response(res)

  def test_video_with_travel(self):
    """ test video with not just general models """

    m = self.app.models.get('travel-v1.0')
    url = 'https://s3.amazonaws.com/samples.clarifai.com/D7qTae7IQLKSI.gif'
    res = m.predict_by_url(url, is_video=True)
    self._verify_video_response(res)


if __name__ == '__main__':
  unittest.main()
