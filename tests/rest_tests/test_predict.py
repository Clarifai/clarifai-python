# -*- coding: utf-8 -*-

import logging
import unittest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Concept
from clarifai.rest import Image as ClImage
from clarifai.rest import ModelOutputInfo, ModelOutputConfig

urls = [
  "https://samples.clarifai.com/metro-north.jpg",
  'https://samples.clarifai.com/3o6gb3kkXfLvdKEZs4.gif',
]


class TestPredict(unittest.TestCase):
  _multiprocess_can_split_ = True
  to_cleanup = []

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(log_level=logging.WARN)

  def test_predict_image_url(self):
    """ predict a single url """

    # just by url
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(urls[0])

  def test_predict_image_url_min_value(self):

    # url, with min_value
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(urls[0], min_value=0.4)
    for c in res['outputs'][0]['data']['concepts']:
      self.assertGreaterEqual(c['value'], 0.4)

  def test_predict_image_url_max_concepts(self):

    # url, with max_concepts
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(urls[0], max_concepts=5)
    self.assertEqual(len(res['outputs'][0]['data']['concepts']), 5)

  def test_predict_image_url_min_value_max_concepts(self):

    # url, with both min_value and max_concepts
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(urls[0], min_value=0.6, max_concepts=5)
    for c in res['outputs'][0]['data']['concepts']:
      self.assertGreaterEqual(c['value'], 0.6)

    self.assertLessEqual(len(res['outputs'][0]['data']['concepts']), 5)

  def test_predict_image_url_select_concepts(self):

    # url, with select_concepts, by name
    m = self.app.models.get('general-v1.3')
    select_concepts = [Concept(concept_name='beer'), Concept(concept_name='vehicle')]
    res = m.predict_by_url(urls[0], select_concepts=select_concepts)
    self.assertEqual(len(res['outputs'][0]['data']['concepts']), 2)

    # url, with select_concepts by id
    select_concepts = [Concept(concept_id='ai_hK1KnTCJ'),
                       Concept(concept_id='ai_m52MdMR3'),
                       Concept(concept_id='ai_fN8NZ9JV')]
    res = m.predict_by_url(urls[0], select_concepts=select_concepts)
    self.assertEqual(len(res['outputs'][0]['data']['concepts']), 3)

  def test_predict_video_url(self):
    """ predict a single url """

    # just by url
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(urls[1], is_video=True)

  def test_predict_video_url_min_value(self):

    # url, with min_value
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(urls[1], is_video=True, min_value=0.9)
    for frame in res['outputs'][0]['data']['frames']:
      for c in frame['data']['concepts']:
        self.assertGreaterEqual(c['value'], 0.9)

  def test_predict_video_url_max_concepts(self):

    # url, with max_concepts
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(urls[1], is_video=True, max_concepts=3)
    for frame in res['outputs'][0]['data']['frames']:
      self.assertEqual(len(frame['data']['concepts']), 3)

  def test_predict_video_url_min_value_max_concepts(self):

    # url, with both min_value and max_concepts
    m = self.app.models.get('general-v1.3')
    res = m.predict_by_url(urls[1], is_video=True, min_value=0.85, max_concepts=3)
    for frame in res['outputs'][0]['data']['frames']:
      for c in frame['data']['concepts']:
        self.assertGreaterEqual(c['value'], 0.85)

    for frame in res['outputs'][0]['data']['frames']:
      self.assertLessEqual(len(frame['data']['concepts']), 3)

  def test_bulk_with_min_value(self):

    img = ClImage(url=urls[0])

    m = self.app.models.get('general-v1.3')
    model_output_info = ModelOutputInfo(output_config=ModelOutputConfig(min_value=0.96))

    res = m.predict(inputs=[img, img, img], model_output_info=model_output_info)

    for result in res['outputs']:
      for c in result['data']['concepts']:
        self.assertGreaterEqual(c['value'], 0.96)


if __name__ == '__main__':
  unittest.main()
