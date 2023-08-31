import dataclasses
import inspect
import logging
import os
import unittest

import numpy as np

from ..model_config.triton_config import TritonModelConfig
from .output import (ClassifierOutput, EmbeddingOutput, ImageOutput, MasksOutput, TextOutput,
                     VisualDetectorOutput)

PREDEFINED_TEXTS = ["Photo of a cat", "A cat is playing around"]

PREDEFINED_IMAGES = [
    np.zeros((100, 100, 3), dtype='uint8'),  #black
    np.ones((100, 100, 3), dtype='uint8') * 255,  #white
    np.random.uniform(0, 255, (100, 100, 3)).astype('uint8')  #noise
]


class DefaultTestInferenceModel(unittest.TestCase):
  """
  This file contains test cases:
  * Test triton config of current model vs default config
  * Test if labels.txt is valid for specific model types
  * Test inference with simple inputs
  ...
  """
  __test__ = False

  def triton_get_predictions(self, input_data):
    """Call InferenceModel.get_predictions method

    Args:
        input_data (Union[np.ndarray, str]):
          if model receives image or vector then type is `np.ndarray`. Otherwise `string`

    Returns:
       One of types in models.output
    """
    return inspect.unwrap(self.triton_model.inference_obj.get_predictions)(
        self.triton_model.inference_obj, input_data)

  def _get_preprocess(self):
    """ preprocess if input is image """
    if "image" in self.triton_model_input_name:
      h, w, _ = self.triton_model_config.input[0].dims
      if h > -1 and w > -1:
        import cv2

        def _f(x):
          logging.info(f"Preprocess reshape image => {(w, h, 3)}")
          return cv2.resize(x, (w, h))

    return None

  def intitialize(
      self,
      model_type: str,
      repo_version_dir: str,
      is_instance_kind_gpu: bool = True,
  ):
    import sys
    sys.path.append(repo_version_dir)
    self.model_type = model_type
    self.is_instance_kind_gpu = is_instance_kind_gpu
    logging.info(self.model_type)
    from model import TritonPythonModel

    # Construct TritonPythonModel object
    self.triton_model = TritonPythonModel()
    self.triton_model.initialize(
        dict(
            model_repository=os.path.join(repo_version_dir, ".."),
            model_instance_kind="GPU" if self.is_instance_kind_gpu else "cpu"))
    # Get default config of model and model_type
    self.default_triton_model_config = TritonModelConfig(
        model_name=self.model_type,
        model_version="1",
        model_type=self.model_type,
        image_shape=[-1, -1])
    # Get current model config
    self.triton_model_config = self.triton_model.config_msg
    self.triton_model_input_name = self.triton_model.input_name
    self.preprocess = self._get_preprocess()
    # load labels
    self._required_label_model_types = [
        "visual-detector", "visual-classifier", "text-classifier", "visual-segmenter"
    ]
    self.labels = []
    if self.model_type in self._required_label_model_types:
      with open(os.path.join(repo_version_dir, "../labels.txt"), 'r') as fp:
        labels = fp.readlines()
        if labels:
          self.labels = [line for line in labels if line]

  def test_triton_config(self):
    """ test Triton config"""
    # check if input names are still matched
    self.assertEqual(
        self.triton_model_input_name, self.default_triton_model_config.input[0].name,
        "input name of current model vs generated model must be matched "
        f"{self.triton_model_input_name} != {self.default_triton_model_config.input[0].name}")
    # check if output names are still matched
    default_output_names = [each.name for each in self.default_triton_model_config.output]
    for output_name in self.triton_model_config.output:
      self.assertIn(output_name.name, default_output_names,
                    "output name of current model vs generated model must be matched "
                    f"{output_name.name} not in {default_output_names}")

  def test_having_labels(self):
    if self.model_type in self._required_label_model_types:
      self.assertTrue(
          len(self.labels),
          f"`labels.txt` is empty!. Model type `{self.model_type}` requires input labels in `labels.txt`"
      )

  def test_inference_with_predefined_inputs(self):
    """ Test Inference with predefined inputs """

    if self.preprocess:
      inputs = [self.preprocess(inp) for inp in PREDEFINED_IMAGES]
    elif "image" in self.triton_model_input_name:
      inputs = PREDEFINED_IMAGES
      logging.info(inputs[0].shape)
    else:
      inputs = PREDEFINED_TEXTS
    outputs = [self.triton_get_predictions(inp) for inp in inputs]

    # Test for specific model type:
    # 1. length of output array vs config
    # 2. type of outputs
    # 3. test range value, shape and dtype of output
    def _is_valid_logit(x: np.array):
      return np.all(0 <= x) and np.all(x <= 1)

    def _is_non_negative(x: np.array):
      return np.all(x >= 0)

    def _is_integer(x):
      return np.all(np.equal(np.mod(x, 1), 0))

    for inp, output in zip(inputs, outputs):

      field = dataclasses.fields(output)[0].name
      self.assertEqual(
          len(self.triton_model_config.output[0].dims),
          len(getattr(output, field).shape),
          "Length of 'dims' of config and output must be matched, but get "
          f"Config {len(self.triton_model_config.output[0].dims)} != Output {len(getattr(output, field).shape)}"
      )

      if self.model_type == "visual-detector":
        logging.info(output.predicted_labels)
        self.assertEqual(
            type(output), VisualDetectorOutput,
            f"Output type must be `VisualDetectorOutput`, but got {type(output)}")
        self.assertTrue(
            _is_valid_logit(output.predicted_scores), "`predicted_scores` must be in range [0, 1]")
        self.assertTrue(
            _is_non_negative(output.predicted_bboxes), "`predicted_bboxes` must be >= 0")
        self.assertTrue(
            np.all(0 <= output.predicted_labels) and
            np.all(output.predicted_labels < len(self.labels)),
            f"`predicted_labels` must be in [0, {len(self.labels) - 1}]")
        self.assertTrue(_is_integer(output.predicted_labels), "`predicted_labels` must be integer")

      elif self.model_type == "visual-classifier":
        self.assertEqual(
            type(output), ClassifierOutput,
            f"Output type must be `ClassifierOutput`, but got {type(output)}")
        self.assertTrue(
            _is_valid_logit(output.predicted_scores), "`predicted_scores` must be in range [0, 1]")
        if self.labels:
          self.assertEqual(
              len(output.predicted_scores),
              len(self.labels),
              f"`predicted_labels` must equal to {len(self.labels)}, however got {len(output.predicted_scores)}"
          )

      elif self.model_type == "text-classifier":
        self.assertEqual(
            type(output), ClassifierOutput,
            f"Output type must be `ClassifierOutput`, but got {type(output)}")
        self.assertTrue(
            _is_valid_logit(output.predicted_scores), "`predicted_scores` must be in range [0, 1]")
        if self.labels:
          self.assertEqual(
              len(output.predicted_scores),
              len(self.labels),
              f"`predicted_labels` must equal to {len(self.labels)}, however got {len(output.predicted_scores)}"
          )

      elif self.model_type == "text-embedder":
        self.assertEqual(
            type(output), EmbeddingOutput,
            f"Output type must be `EmbeddingOutput`, but got {type(output)}")
        self.assertNotEqual(output.embedding_vector.shape, [])

      elif self.model_type == "text-to-text":
        self.assertEqual(
            type(output), TextOutput, f"Output type must be `TextOutput`, but got {type(output)}")

      elif self.model_type == "text-to-image":
        self.assertEqual(
            type(output), ImageOutput,
            f"Output type must be `ImageOutput`, but got {type(output)}")
        self.assertTrue(_is_non_negative(output.image), "`image` elements must be >= 0")

      elif self.model_type == "visual-embedder":
        self.assertEqual(
            type(output), EmbeddingOutput,
            f"Output type must be `EmbeddingOutput`, but got {type(output)}")
        self.assertNotEqual(output.embedding_vector.shape, [])

      elif self.model_type == "visual-segmenter":
        self.assertEqual(
            type(output), MasksOutput,
            f"Output type must be `MasksOutput`, but got {type(output)}")
        self.assertTrue(_is_integer(output.predicted_mask), "`predicted_mask` must be integer")
        if self.labels:
          self.assertTrue(
              np.all(0 <= output.predicted_mask) and
              np.all(output.predicted_mask < len(self.labels)),
              f"`predicted_mask` must be in [0, {len(self.labels) - 1}]")


if __name__ == '__main__':
  unittest.main()
