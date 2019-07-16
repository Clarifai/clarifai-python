# -*- coding: utf-8 -*-

import unittest

from clarifai.rest import ClarifaiApp, Workflow

from . import sample_inputs


class TestWorkflows(unittest.TestCase):
  """
  unit test for workflow related calls
  """

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(quiet=True)

  def test_list_and_get_workflows(self):
    for wf in self.app.workflows.get_all():
      self.assertTrue(isinstance(wf, Workflow))
      wf_get = self.app.workflows.get(wf.wf_id)
      self.assertEqual(wf_get.wf_id, wf.wf_id)

  def test_list_and_get_public_workflows(self):
    for wf in self.app.workflows.get_all(public_only=True):
      self.assertTrue(isinstance(wf, Workflow))
      wf_get = self.app.workflows.get(wf.wf_id)
      self.assertEqual(wf_get.wf_id, wf.wf_id)

  def test_public_workflow_predict_by_url(self):

    wf = self.app.workflows.get('General')
    res = wf.predict_by_url(sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(10000, res['status']['code'])

  def test_public_workflow_predict_by_filename(self):

    wf = self.app.workflows.get('General')
    res = wf.predict_by_filename(sample_inputs.METRO_IMAGE_FILE_PATH)
    self.assertEqual(10000, res['status']['code'])

  def test_predict_workflow(self):
    wf = self.app.workflows.get('food-and-general')
    res = wf.predict_by_filename(sample_inputs.METRO_IMAGE_FILE_PATH)
    self.assertEqual(10000, res['status']['code'])
