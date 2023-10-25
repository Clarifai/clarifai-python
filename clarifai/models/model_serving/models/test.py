import logging
import os
import unittest

from clarifai.models.model_serving.models.default_test import DefaultTestInferenceModel


class CustomTestInferenceModel(DefaultTestInferenceModel):
  """
  Run this file to test your implementation of InferenceModel in inference.py with default tests of Triton configuration and its output values based on basic predefined inputs
  If you want to write custom testcase or just test output value.
  Please follow these instrucitons:
  1. Name your test function with prefix "test" so that pytest can execute
  2. In order to obtain output of InferenceModel, call `self.triton_get_predictions(input_data)`.
  3. If your input is `image` and you have set custom size of it when building model repository,
  call `self.preprocess(image)` to obtain correct resized input
  4. Run this test by calling
  ```bash
  pytest ./your_triton_folder/1/test.py
  #to see std output
  pytest --log-cli-level=INFO  -s ./your_triton_folder/1/test.py
  ```

  ### Examples:
  + test text-to-image output
  ```
  def test_text_to_image_output(self):
    text = "Test text"
    output = self.triton_get_predictions(text)
    image = output.image # uint8 np.ndarray image
    #show or save
  ```
  + test visual-classifier output
  ```
  def test_visual_classifier(self):
    image = cv2.imread("your/local/image.jpg") # Keep in mind of format of image (BGR or RGB)
    output = self.triton_get_predictions(image)
    scores = output.predicted_scores # np.ndarray
    #process scores to get class id and its score
    logger.info(result)
  """

  # Insert your inference parameters json path here
  # or insert a dictionary of your_parameter_name and value, e.g dict(x=1.5, y="text", c=True)
  # or Leave it as "" if you don't have it.
  inference_parameters = ""

  ########### Initialization. Do not change it ###########
  __test__ = True

  def setUp(self) -> None:
    logging.info("Initializing...")
    model_type = "clarifai-model-type"  # your model type
    self.intitialize(
        model_type,
        repo_version_dir=os.path.dirname(__file__),
        is_instance_kind_gpu=True,
        inference_parameters=self.inference_parameters)

  ########################################################


if __name__ == '__main__':
  unittest.main()
