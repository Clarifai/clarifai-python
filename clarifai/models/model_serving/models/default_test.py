import dataclasses
import inspect
import logging
import os
import unittest
from typing import Any, Dict, Union

import numpy as np

from ..model_config import ModelTypes
from ..model_config.config import get_model_config
from ..model_config.inference_parameter import InferParamManager
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

  def triton_get_predictions(self, input_data, **kwargs):
    """Call InferenceModel.get_predictions method

    Args:
        input_data (Union[np.ndarray, str]):
          if model receives image or vector then type is `np.ndarray`, otherwise `string`

    Returns:
       One of types in models.output
    """
    _kwargs = self.inference_parameters.validate(**kwargs)
    return inspect.unwrap(self.triton_python_model.inference_obj.get_predictions)(
        self.triton_python_model.inference_obj, input_data, **_kwargs)

  def _get_preprocess(self, input):
    """ preprocess if input is image """
    if "image" in input.name:
      h, w, _ = input.dims
      if h > -1 and w > -1:
        import cv2

        def _f(x):
          logging.info(f"Preprocess reshape image => {(w, h, 3)}")
          return cv2.resize(x, (w, h))

        return _f

    return lambda x: x

  def intitialize(self,
                  model_type: str,
                  repo_version_dir: str,
                  is_instance_kind_gpu: bool = True,
                  inference_parameters: Union[str, Dict[str, Any]] = ""):
    import sys
    sys.path.append(repo_version_dir)
    self.model_type = model_type
    self.is_instance_kind_gpu = is_instance_kind_gpu
    logging.info(self.model_type)

    # load inference parameters
    if isinstance(inference_parameters, str):
      self.inference_parameters = InferParamManager(json_path=inference_parameters)
    else:
      self.inference_parameters = InferParamManager.from_kwargs(**inference_parameters)
      exported_file_path = os.path.join(repo_version_dir, "inference_parameters.json")
      logging.info(f"Export inference parameters to `{exported_file_path}` when loading from dict")
      self.inference_parameters.export(exported_file_path)

    # Construct TritonPythonModel object
    from model import TritonPythonModel
    self.triton_python_model = TritonPythonModel()
    self.triton_python_model.initialize(
        dict(
            model_repository=os.path.join(repo_version_dir, ".."),
            model_instance_kind="GPU" if self.is_instance_kind_gpu else "cpu"))
    # Get default config of model and model_type
    self.default_triton_model_config = get_model_config(self.model_type).make_triton_model_config(
        model_name=self.model_type, model_version="1", image_shape=[-1, -1])
    # Get current model config
    self.triton_model_config = self.triton_python_model.config_msg
    self.input_name_to_config = {each.name: each
                                 for each in self.triton_model_config.input}  # name: input
    self.preprocess = {
        k: self._get_preprocess(input)
        for k, input in self.input_name_to_config.items()
    }
    # load labels
    self._required_label_model_types = [
        ModelTypes.visual_detector, ModelTypes.visual_classifier, ModelTypes.text_classifier,
        ModelTypes.visual_segmenter
    ]
    self._output_text_models = [ModelTypes.text_to_text]
    self.labels = []
    if self.model_type in self._required_label_model_types:
      with open(os.path.join(repo_version_dir, "../labels.txt"), 'r') as fp:
        labels = fp.readlines()
        if labels:
          self.labels = [line for line in labels if line]

  def test_triton_config(self):
    """ test Triton config"""
    # check if input names are still matched
    default_input_names = [each.name for each in self.default_triton_model_config.input]
    current_input_names = [each.name for each in self.triton_model_config.input]
    default_input_names.sort()
    current_input_names.sort()
    self.assertEqual(current_input_names, default_input_names,
                     "input name of current model vs generated model must be matched "
                     f"{current_input_names} != {default_input_names}")
    # check if output names are still matched
    default_output_names = [each.name for each in self.default_triton_model_config.output]
    current_output_names = [each.name for each in self.triton_model_config.output]
    default_output_names.sort()
    current_output_names.sort()
    self.assertEqual(current_output_names, default_output_names,
                     "output name of current model vs generated model must be matched "
                     f"{current_output_names} not in {default_output_names}")

  def test_having_labels(self):
    if self.model_type in self._required_label_model_types:
      self.assertTrue(
          len(self.labels),
          f"`labels.txt` is empty!. Model type `{self.model_type}` requires input labels in `labels.txt`"
      )

  def test_inference_with_predefined_inputs(self):
    """ Test Inference with predefined inputs """

    def _is_valid_logit(x: np.array):
      return np.all(0 <= x) and np.all(x <= 1)

    def _is_non_negative(x: np.array):
      return np.all(x >= 0)

    def _is_integer(x):
      return np.all(np.equal(np.mod(x, 1), 0))

    if len(self.input_name_to_config) == 1:
      if "image" in self.preprocess:
        inputs = [self.preprocess["image"](inp) for inp in PREDEFINED_IMAGES]
      else:
        inputs = PREDEFINED_TEXTS
      outputs = [self.triton_get_predictions(inp) for inp in inputs]

      # Test for specific model type:
      # 1. length of output array vs config
      # 2. type of outputs
      # 3. test range value, shape and dtype of output

      for inp, output in zip(inputs, outputs):

        field = dataclasses.fields(output)[0].name
        if self.model_type not in self._output_text_models:
          self.assertEqual(
              len(self.triton_model_config.output[0].dims),
              len(getattr(output, field).shape),
              "Length of 'dims' of config and output must be matched, but get "
              f"Config {len(self.triton_model_config.output[0].dims)} != Output {len(getattr(output, field).shape)}"
          )

        if self.model_type == ModelTypes.visual_detector:
          logging.info(output.predicted_labels)
          self.assertEqual(
              type(output), VisualDetectorOutput,
              f"Output type must be `VisualDetectorOutput`, but got {type(output)}")
          self.assertTrue(
              _is_valid_logit(output.predicted_scores),
              "`predicted_scores` must be in range [0, 1]")
          self.assertTrue(
              _is_non_negative(output.predicted_bboxes), "`predicted_bboxes` must be >= 0")
          self.assertTrue(
              np.all(0 <= output.predicted_labels) and
              np.all(output.predicted_labels < len(self.labels)),
              f"`predicted_labels` must be in [0, {len(self.labels) - 1}]")
          self.assertTrue(
              _is_integer(output.predicted_labels), "`predicted_labels` must be integer")

        elif self.model_type == ModelTypes.visual_classifier:
          self.assertEqual(
              type(output), ClassifierOutput,
              f"Output type must be `ClassifierOutput`, but got {type(output)}")
          self.assertTrue(
              _is_valid_logit(output.predicted_scores),
              "`predicted_scores` must be in range [0, 1]")
          if self.labels:
            self.assertEqual(
                len(output.predicted_scores),
                len(self.labels),
                f"`predicted_labels` must equal to {len(self.labels)}, however got {len(output.predicted_scores)}"
            )

        elif self.model_type == ModelTypes.text_classifier:
          self.assertEqual(
              type(output), ClassifierOutput,
              f"Output type must be `ClassifierOutput`, but got {type(output)}")
          self.assertTrue(
              _is_valid_logit(output.predicted_scores),
              "`predicted_scores` must be in range [0, 1]")
          if self.labels:
            self.assertEqual(
                len(output.predicted_scores),
                len(self.labels),
                f"`predicted_labels` must equal to {len(self.labels)}, however got {len(output.predicted_scores)}"
            )

        elif self.model_type == ModelTypes.text_embedder:
          self.assertEqual(
              type(output), EmbeddingOutput,
              f"Output type must be `EmbeddingOutput`, but got {type(output)}")
          self.assertNotEqual(output.embedding_vector.shape, [])

        elif self.model_type == ModelTypes.text_to_text:
          self.assertEqual(
              type(output), TextOutput,
              f"Output type must be `TextOutput`, but got {type(output)}")

        elif self.model_type == ModelTypes.text_to_image:
          self.assertEqual(
              type(output), ImageOutput,
              f"Output type must be `ImageOutput`, but got {type(output)}")
          self.assertTrue(_is_non_negative(output.image), "`image` elements must be >= 0")

        elif self.model_type == ModelTypes.visual_embedder:
          self.assertEqual(
              type(output), EmbeddingOutput,
              f"Output type must be `EmbeddingOutput`, but got {type(output)}")
          self.assertNotEqual(output.embedding_vector.shape, [])

        elif self.model_type == ModelTypes.visual_segmenter:
          self.assertEqual(
              type(output), MasksOutput,
              f"Output type must be `MasksOutput`, but got {type(output)}")
          self.assertTrue(_is_integer(output.predicted_mask), "`predicted_mask` must be integer")
          if self.labels:
            self.assertTrue(
                np.all(0 <= output.predicted_mask) and
                np.all(output.predicted_mask < len(self.labels)),
                f"`predicted_mask` must be in [0, {len(self.labels) - 1}]")

    elif len(self.input_name_to_config) == 2:
      from itertools import zip_longest
      if self.model_type == ModelTypes.multimodal_embedder:
        input_images = [self.preprocess["image"](inp) for inp in PREDEFINED_IMAGES]
        input_texts = PREDEFINED_TEXTS

        def _assert(input_data):
          for group in zip_longest(*input_data.values()):
            _input = dict(zip(input_data, group))
            output = self.triton_get_predictions(input_data=_input)
            self.assertEqual(
                type(output), EmbeddingOutput,
                f"Output type must be `EmbeddingOutput`, but got {type(output)}")
          self.assertNotEqual(output.embedding_vector.shape, [])

        _assert(dict(image=input_images, text=[]))
        _assert(dict(image=[], text=input_texts))


if __name__ == '__main__':
  unittest.main()
