# -*- coding: utf-8 -*-

import unittest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Workflow

urls = [
  "https://samples.clarifai.com/metro-north.jpg",
  "https://samples.clarifai.com/wedding.jpg",
]


class TestWorkflows(unittest.TestCase):
  """
  unit test for workflow related calls
  """

  _multiprocess_can_split_ = True

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(quiet=True)

  def test_public_workflow(self):

    for wf in self.app.workflows.get_all(public_only=True):
      self.assertTrue(isinstance(wf, Workflow))
      wf_get = self.app.workflows.get(wf.wf_id)
      self.assertEqual(wf_get.wf_id, wf.wf_id)

  def test_public_workflow_predict(self):

    """
    for wf in self.app.workflows.get_all(public_only=True):
      self.assertTrue(isinstance(wf, Workflow))
      wf_get = self.app.workflows.get(wf.wf_id)
      wf_get.predict_by_url(urls[0])
    """
    wf = self.app.workflows.get('General')
    res = wf.predict_by_url(urls[0])

  def test_list_workflow(self):

    for wf in self.app.workflows.get_all():
      pass

  def get_workflow(self):
    pass

  def test_predict_workflow(self):
    pass

if __name__ == '__main__':
  unittest.main()
