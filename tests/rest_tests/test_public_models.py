# -*- coding: utf-8 -*-

import logging
import unittest

from clarifai.rest import ClarifaiApp

from . import sample_inputs


class TestPublicModels(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(log_level=logging.WARN)

  def test_predict_with_apparel_model(self):
    res = self.app.public_models.apparel_model.predict_by_url(url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_celebrity_model(self):
    res = self.app.public_models.celebrity_model.predict_by_url(url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_color_mode(self):
    res = self.app.public_models.color_model.predict_by_url(url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_demographics_mode(self):
    res = self.app.public_models.demographics_model.predict_by_url(
        url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_face_detection_model(self):
    res = self.app.public_models.face_detection_model.predict_by_url(
        url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_face_embedding_model(self):
    res = self.app.public_models.face_embedding_model.predict_by_url(
        url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_focus_model(self):
    res = self.app.public_models.focus_model.predict_by_url(url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_food_model(self):
    res = self.app.public_models.food_model.predict_by_url(url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_general_embedding_model(self):
    res = self.app.public_models.general_embedding_model.predict_by_url(
        url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_general_model(self):
    res = self.app.public_models.general_model.predict_by_url(url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_landscape_quality_model(self):
    res = self.app.public_models.landscape_quality_model.predict_by_url(
        url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_logo_model(self):
    res = self.app.public_models.logo_model.predict_by_url(url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_moderation_model(self):
    res = self.app.public_models.moderation_model.predict_by_url(url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_nsfw_model(self):
    res = self.app.public_models.nsfw_model.predict_by_url(url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_portrait_model(self):
    res = self.app.public_models.portrait_quality_model.predict_by_url(
        url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_textures_and_patterns_model(self):
    res = self.app.public_models.textures_and_patterns_model.predict_by_url(
        url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_travel_model(self):
    res = self.app.public_models.travel_model.predict_by_url(url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_wedding_model(self):
    res = self.app.public_models.wedding_model.predict_by_url(url=sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_with_segmentation_model(self):
    # only runs this test in prod as the segmentation models only exist in prod
    if self.app.api.base_url == 'api.clarifai.com':
      # test people segmentation model
      m = self.app.models.get(model_id='9bef4f02cb40e4709fa254d597b2942d')
      m.model_version = 'db92abe2e6714d7992b359c1fc269ce5'
      res = m.predict_by_url(sample_inputs.METRO_IMAGE_URL)

      # ensure that only 'person' concepts are returned
      # this tests for the correct filtering of concepts in the backend
      for region in res['outputs'][0]['data']['regions']:
        for concept in region['data']['concepts']:
          self.assertEqual('person', concept['name'])

      # test car segmentation
      m = self.app.models.get(model_id='b17ab715bba448228ef916caaaf8d35b')
      m.model_version = 'fc92eca9d7cb45cc8832ab40ddc95c3c'
      res = m.predict_by_url(sample_inputs.METRO_IMAGE_URL)

      # ensure that only 'car' concepts are returned
      for region in res['outputs'][0]['data']['regions']:
        for concept in region['data']['concepts']:
          self.assertEqual('car', concept['name'])
