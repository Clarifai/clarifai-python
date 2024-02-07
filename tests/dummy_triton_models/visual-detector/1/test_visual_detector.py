import unittest

from clarifai.models.model_serving.repo_build import BaseTest


class CustomTest(unittest.TestCase):
  """
  BaseTest loads the InferenceModel from the inference.py file in the current working directory.
  To execute the predict method of the InferenceModel, use the predict method in BaseTest.
  It takes the exact same inputs and inference parameters, returning the same outputs as InferenceModel.predict.
  The difference is that BaseTest.predict verifies your_infer_parameters against config.clarifai_models.inference_parameters and checks the output values.

  For example, test input value of visual-classifier

  def test_input(self):
    import cv2
    path = "path/to/image"
    img = cv2.imread(path)
    outputs = self.model.predict([img], infer_param1=..., infer_param2=...)
    print(outputs)
    assert outputs

  """

  def setUp(self) -> None:
    your_infer_parameter = dict(
    )  # for example dict(float_var=0.12, string_var="test", _secret_string_var="secret")
    self.model = BaseTest(your_infer_parameter)

  def test_default_cases(self):
    """Test your model with dummy inputs.
    In general, you only need to run this test to check your InferneceModel implementation.
    In case the default inputs makes your model failed for some reason (not because of assert in `test_with_default_inputs`),
    you can comment out this test.
    """
    self.model.test_with_default_inputs()

  def test_specific_case1(self):
    """ Implement your test case"""
    pass
