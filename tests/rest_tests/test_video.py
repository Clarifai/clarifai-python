# -*- coding: utf-8 -*-

import base64
import logging
import os
import tempfile
import unittest

from clarifai.rest import ClarifaiApp
from clarifai.rest import Video as ClVideo

from . import sample_inputs


class TestVideo(unittest.TestCase):
  """
  unit test for video related calls
  """

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(log_level=logging.WARN)
    cls.m = cls.app.models.get('general-v1.3')

  def test_create_video_obj(self):
    """ test creating Clarifai Video object from different sources """
    v1 = ClVideo(url=sample_inputs.CONAN_GIF_VIDEO_URL)

    self.assertTrue(isinstance(v1, ClVideo))
    self.assertEqual(v1.url, sample_inputs.CONAN_GIF_VIDEO_URL)

    v1_json = v1.dict()
    self.assertIn('video', v1_json['data'])
    self.assertIn('url', v1_json['data']['video'])

    # test url with leading or trailing spaces
    v1 = ClVideo(url=' ' + sample_inputs.CONAN_GIF_VIDEO_URL)
    self.assertEqual(v1.url, sample_inputs.CONAN_GIF_VIDEO_URL)
    v1 = ClVideo(url=' ' + sample_inputs.CONAN_GIF_VIDEO_URL + '  ')
    self.assertEqual(v1.url, sample_inputs.CONAN_GIF_VIDEO_URL)

    with open(sample_inputs.SMALL_VIDEO_FILE_PATH, 'rb') as f:
      video_file_bytes = f.read()

    # test video file
    v2 = ClVideo(filename=sample_inputs.SMALL_VIDEO_FILE_PATH)

    toddler_flowers_base64_bytes = base64.b64encode(video_file_bytes)
    v3 = ClVideo(base64=toddler_flowers_base64_bytes)

    with open(sample_inputs.SMALL_VIDEO_FILE_PATH, 'rb') as f:
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

    res = self.m.predict_by_url(sample_inputs.CONAN_GIF_VIDEO_URL, is_video=True)
    self._verify_video_response(res)

    res = self.m.predict_by_url(sample_inputs.BEER_VIDEO_URL, is_video=True)
    self._verify_video_response(res)

  def test_predict_with_raw_bytes(self):
    """ test video predict with other sources rather than url """

    # predict by file raw bytes
    raw_bytes = self.app.api.session.get(sample_inputs.SMALL_GIF_VIDEO_URL).content
    res = self.m.predict_by_bytes(raw_bytes, is_video=True)
    self._verify_video_response(res)

  def test_predict_with_base64_bytes(self):
    raw_bytes = self.app.api.session.get(sample_inputs.SMALL_GIF_VIDEO_URL).content
    base64_bytes = base64.b64encode(raw_bytes)
    res = self.m.predict_by_base64(base64_bytes, is_video=True)
    self._verify_video_response(res)

  def test_predict_with_file(self):
    raw_bytes = self.app.api.session.get(sample_inputs.SMALL_GIF_VIDEO_URL).content
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
    res = m.predict_by_url(sample_inputs.SMALL_GIF_VIDEO_URL, is_video=True)
    self._verify_video_response(res)

  def test_video_with_nsfw(self):
    """ test video with not just general models """

    m = self.app.models.get('nsfw-v1.0')
    res = m.predict_by_url(sample_inputs.SMALL_GIF_VIDEO_URL, is_video=True)
    self._verify_video_response(res)

  def test_video_with_wedding(self):
    """ test video with not just general models """

    m = self.app.models.get('weddings-v1.0')
    res = m.predict_by_url(sample_inputs.SMALL_GIF_VIDEO_URL, is_video=True)
    self._verify_video_response(res)

  def test_video_with_food(self):
    """ test video with not just general models """

    m = self.app.models.get('food-items-v1.0')
    res = m.predict_by_url(sample_inputs.SMALL_GIF_VIDEO_URL, is_video=True)
    self._verify_video_response(res)

  def test_video_with_travel(self):
    """ test video with not just general models """

    m = self.app.models.get('travel-v1.0')
    res = m.predict_by_url(sample_inputs.SMALL_GIF_VIDEO_URL, is_video=True)
    self._verify_video_response(res)

  def test_predict_video_url_with_custom_sample_ms(self):
    """ test video with not just general models """

    m = self.app.public_models.general_model
    res = m.predict_by_url(sample_inputs.BEER_VIDEO_URL, is_video=True, sample_ms=2000)
    self._verify_video_response(res)
    for frame in res['outputs'][0]['data']['frames']:
      assert frame['frame_info']['time'] % 2000 == 0

  def test_predict_video_filename_with_custom_sample_ms(self):
    """ test video with not just general models """

    m = self.app.public_models.general_model
    res = m.predict_by_filename(sample_inputs.BEER_VIDEO_FILE_PATH, is_video=True, sample_ms=2000)
    self._verify_video_response(res)
    for frame in res['outputs'][0]['data']['frames']:
      assert frame['frame_info']['time'] % 2000 == 0

  def test_predict_video_bytes_with_custom_sample_ms(self):
    """ test video with not just general models """

    m = self.app.public_models.general_model
    file_bytes = open(sample_inputs.BEER_VIDEO_FILE_PATH, 'rb').read()
    res = m.predict_by_bytes(file_bytes, is_video=True, sample_ms=2000)
    self._verify_video_response(res)
    for frame in res['outputs'][0]['data']['frames']:
      assert frame['frame_info']['time'] % 2000 == 0

  def test_predict_video_base64_with_custom_sample_ms(self):
    """ test video with not just general models """

    m = self.app.public_models.general_model
    file_bytes = open(sample_inputs.BEER_VIDEO_FILE_PATH, 'rb').read()
    res = m.predict_by_base64(base64.encodestring(file_bytes), is_video=True, sample_ms=2000)
    self._verify_video_response(res)
    for frame in res['outputs'][0]['data']['frames']:
      assert frame['frame_info']['time'] % 2000 == 0
