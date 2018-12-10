# -*- coding: utf-8 -*-

import base64
import logging
import unittest

from clarifai.rest import ClarifaiApp, Concept
from clarifai.rest import Image as ClImage
from clarifai.rest import ModelOutputConfig, ModelOutputInfo

from . import sample_inputs


class TestPredict(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(log_level=logging.WARN)

  def test_predict_image_url(self):
    """ predict a single url """

    # just by url
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])
    assert 'railway' in [concept['name'] for concept in res['outputs'][0]['data']['concepts']]

  def test_predict_image_url_min_value(self):

    # url, with min_value
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(sample_inputs.METRO_IMAGE_URL, min_value=0.4)
    self.assertEqual(10000, res['status']['code'])
    for c in res['outputs'][0]['data']['concepts']:
      self.assertGreaterEqual(c['value'], 0.4)

  def test_predict_image_url_max_concepts(self):

    # url, with max_concepts
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(sample_inputs.METRO_IMAGE_URL, max_concepts=5)
    self.assertEqual(10000, res['status']['code'])
    self.assertEqual(len(res['outputs'][0]['data']['concepts']), 5)

  def test_predict_image_url_min_value_max_concepts(self):

    # url, with both min_value and max_concepts
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(sample_inputs.METRO_IMAGE_URL, min_value=0.6, max_concepts=5)
    self.assertEqual(10000, res['status']['code'])
    for c in res['outputs'][0]['data']['concepts']:
      self.assertGreaterEqual(c['value'], 0.6)

    self.assertLessEqual(len(res['outputs'][0]['data']['concepts']), 5)

  def test_predict_image_url_select_concepts(self):

    # url, with select_concepts, by name
    m = self.app.models.get('general-v1.3')
    select_concepts = [Concept(concept_name='beer'), Concept(concept_name='vehicle')]
    res = m.predict_by_url(sample_inputs.METRO_IMAGE_URL, select_concepts=select_concepts)
    self.assertEqual(10000, res['status']['code'])
    self.assertEqual(len(res['outputs'][0]['data']['concepts']), 2)

    # url, with select_concepts by id
    select_concepts = [
        Concept(concept_id='ai_hK1KnTCJ'),
        Concept(concept_id='ai_m52MdMR3'),
        Concept(concept_id='ai_fN8NZ9JV')
    ]
    res = m.predict_by_url(sample_inputs.METRO_IMAGE_URL, select_concepts=select_concepts)
    self.assertEqual(10000, res['status']['code'])
    self.assertEqual(len(res['outputs'][0]['data']['concepts']), 3)

  def test_predict_video_url(self):
    """ predict a single url """

    # just by url
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(sample_inputs.CONAN_GIF_VIDEO_URL, is_video=True)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_video_url_min_value(self):

    # url, with min_value
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(sample_inputs.CONAN_GIF_VIDEO_URL, is_video=True, min_value=0.9)
    self.assertEqual(10000, res['status']['code'])
    for frame in res['outputs'][0]['data']['frames']:
      for c in frame['data']['concepts']:
        self.assertGreaterEqual(c['value'], 0.9)

  def test_predict_video_url_max_concepts(self):

    # url, with max_concepts
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(sample_inputs.CONAN_GIF_VIDEO_URL, is_video=True, max_concepts=3)
    self.assertEqual(10000, res['status']['code'])
    for frame in res['outputs'][0]['data']['frames']:
      self.assertEqual(len(frame['data']['concepts']), 3)

  def test_predict_video_url_min_value_max_concepts(self):

    # url, with both min_value and max_concepts
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(
        sample_inputs.CONAN_GIF_VIDEO_URL, is_video=True, min_value=0.85, max_concepts=3)
    self.assertEqual(10000, res['status']['code'])
    for frame in res['outputs'][0]['data']['frames']:
      for c in frame['data']['concepts']:
        self.assertGreaterEqual(c['value'], 0.85)

    for frame in res['outputs'][0]['data']['frames']:
      self.assertLessEqual(len(frame['data']['concepts']), 3)

  def test_bulk_with_min_value(self):

    img = ClImage(url=sample_inputs.METRO_IMAGE_URL)

    m = self.app.models.get('general-v1.3')
    model_output_info = ModelOutputInfo(output_config=ModelOutputConfig(min_value=0.96))

    res = m.predict(inputs=[img, img, img], model_output_info=model_output_info)
    self.assertEqual(10000, res['status']['code'])

    for result in res['outputs']:
      for c in result['data']['concepts']:
        self.assertGreaterEqual(c['value'], 0.96)

  def test_predict_image_by_filename(self):
    """ predict a an image by file name """

    m = self.app.models.get('general-v1.3')
    res = m.predict_by_filename(sample_inputs.METRO_IMAGE_FILE_PATH)
    self.assertEqual(10000, res['status']['code'])
    assert 'railway' in [concept['name'] for concept in res['outputs'][0]['data']['concepts']]

  def test_predict_image_by_bytes(self):
    """ predict a an image by file name """

    with open(sample_inputs.METRO_IMAGE_FILE_PATH, 'rb') as f:
      image_bytes = f.read()

    m = self.app.models.get('general-v1.3')
    res = m.predict_by_bytes(image_bytes)
    self.assertEqual(10000, res['status']['code'])
    assert 'railway' in [concept['name'] for concept in res['outputs'][0]['data']['concepts']]

  def test_predict_image_by_base64(self):
    """ predict a an image by file name """

    with open(sample_inputs.METRO_IMAGE_FILE_PATH, 'rb') as f:
      image_base64 = base64.b64encode(f.read())

    m = self.app.models.get('general-v1.3')
    res = m.predict_by_base64(image_base64)
    self.assertEqual(10000, res['status']['code'])
    assert 'railway' in [concept['name'] for concept in res['outputs'][0]['data']['concepts']]
