# -*- coding: utf-8 -*-

import unittest

from clarifai.rest import ClarifaiApp, Workflow

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

  def test_public_workflow_predict(self):

    wf = self.app.workflows.get('General')
    res = wf.predict_by_url(urls[0])
    self.assertEqual(10000, res['status']['code'])

  def test_predict_workflow(self):
    # TODO(Rok) MEDIUM: Make a custom workflow so this test can be written.
    pass


if __name__ == '__main__':
  unittest.main()
