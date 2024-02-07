import os
from copy import deepcopy
from typing import Dict, Iterable, List, Union

import numpy as np
import yaml

from ...constants import IMAGE_TENSOR_NAME, TEXT_TENSOR_NAME
from ...model_config import (ClassifierOutput, EmbeddingOutput, ImageOutput, InferParam,
                             InferParamManager, MasksOutput, ModelTypes, TextOutput,
                             VisualDetector, load_user_config)

_default_texts = ["Photo of a cat", "A cat is playing around", "Hello, this is test"]

_default_images = [
    np.zeros((100, 100, 3), dtype='uint8'),  #black
    np.ones((100, 100, 3), dtype='uint8') * 255,  #white
    np.random.uniform(0, 255, (100, 100, 3)).astype('uint8')  #noise
]


def _is_valid_logit(x: np.array):
  return np.all(0 <= x) and np.all(x <= 1)


def _is_non_negative(x: np.array):
  return np.all(x >= 0)


def _is_integer(x):
  return np.all(np.equal(np.mod(x, 1), 0))


class BaseTest:
  init_inference_parameters = {}

  def __init__(self, init_inference_parameters={}) -> None:
    import sys
    if 'inference' in sys.modules:
      del sys.modules['inference']
    import inference
    from inference import InferenceModel
    self.model = InferenceModel()
    self._base_dir = os.path.dirname(inference.__file__)
    self.cfg_path = os.path.join(self._base_dir, "clarifai_config.yaml")
    self.user_config = load_user_config(self.cfg_path)
    self._user_labels = None
    # check if labels exists
    for output_config in self.user_config.serving_backend.triton.output:
      if output_config.label_filename:
        self._user_labels = self.user_config.clarifai_model.labels
        assert self._user_labels, f"Model type `{self.user_config.clarifai_model.type}` requires labels, "\
        f"but can not found value of `clarifai_model.labels` in {self.cfg_path}. Please update this attribute to build the model"

    # update init vs user_defined params
    user_defined_infer_params = [
        InferParam(**each) for each in self.user_config.clarifai_model.inference_parameters
    ]
    total_infer_params = []
    if init_inference_parameters:
      self.init_inference_parameters = init_inference_parameters
    for k, v in self.init_inference_parameters.items():
      _exist = False
      for user_param in user_defined_infer_params:
        if user_param.path == k:
          if user_param.default_value != v:
            print(f"Warning: Overwrite parameter `{k}` with default value `{v}`")
          user_param.default_value = v
          _exist = True
          total_infer_params.append(user_param)
          user_defined_infer_params.remove(user_param)
          break
      if not _exist:
        total_infer_params.append(InferParamManager.from_kwargs(**{k: v}).params[0])

    self.infer_param_manager = InferParamManager(
        params=total_infer_params + user_defined_infer_params)
    self.user_config.clarifai_model.inference_parameters = self.infer_param_manager.get_list_params(
    )
    self._overwrite_cfg()

  @property
  def user_labels(self):
    return self._user_labels

  def _overwrite_cfg(self):
    config = yaml.dump(self.user_config.dump_to_user_config(),)
    with open(self.cfg_path, "w") as f:
      f.write(config)

  def predict(self, input_data: Union[List[np.ndarray], List[str], Dict[str, Union[List[
      np.ndarray], List[str]]]], **inference_parameters) -> Iterable:
    """
    Test Prediction method is exact `InferenceModel.predict` method with
    checking inference paramters.

    Args:
    -----
    - input_data: A list of input data item to predict on. The type depends on model input type:
      * `image`: List[np.ndarray]
      * `text`: List[str]
      * `multimodal`:
        input_data is list of dict where key is input type name e.i. `image`, `text` and value is list.
        {"image": List[np.ndarray], "text": List[str]}

    - **inference_parameters: keyword args of your inference parameters.

    Returns:
    --------
      List of your inference model output type
    """
    infer_params = self.infer_param_manager.validate(**inference_parameters)
    outputs = self.model.predict(input_data=input_data, inference_parameters=infer_params)
    outputs = self._verify_outputs(outputs)
    return outputs

  def _verify_outputs(self, outputs: List[Union[ClassifierOutput, VisualDetector, EmbeddingOutput,
                                                TextOutput, ImageOutput, MasksOutput]]):
    """Test output value/dims

    Args:
      outputs (List[Union[ClassifierOutput, VisualDetector, EmbeddingOutput, TextOutput, ImageOutput, MasksOutput]]): Outputs of `predict` method
    """
    _outputs = deepcopy(outputs)
    _output = _outputs[0]

    if isinstance(_output, EmbeddingOutput):
      # not test
      pass
    elif isinstance(_output, ClassifierOutput):
      for each in _outputs:
        assert _is_valid_logit(each.predicted_scores), "`predicted_scores` must be in range [0, 1]"
        assert len(each.predicted_scores) == len(
            self.user_labels
        ), f"`predicted_scores` dim must be equal to labels, got {len(each.predicted_scores)} != labels {len(self.user_labels)}"
    elif isinstance(_output, VisualDetector):
      for each in _outputs:
        assert _is_valid_logit(each.predicted_scores), "`predicted_scores` must be in range [0, 1]"
        assert _is_integer(each.predicted_labels), "`predicted_labels` must be integer"
        assert np.all(0 <= each.predicted_labels) and np.all(each.predicted_labels < len(
            self.user_labels)), f"`predicted_labels` must be in [0, {len(self.user_labels) - 1}]"
        assert _is_non_negative(each.predicted_bboxes), "`predicted_bboxes` must be >= 0"
    elif isinstance(_output, MasksOutput):
      for each in _outputs:
        assert np.all(0 <= each.predicted_mask) and np.all(each.predicted_mask < len(
            self.user_labels)), f"`predicted_mask` must be in [0, {len(self.user_labels) - 1}]"
    elif isinstance(_output, TextOutput):
      pass
    elif isinstance(_output, ImageOutput):
      for each in _outputs:
        assert _is_non_negative(each.image), "`image` must be >= 0"
    else:
      pass

    return outputs

  def test_with_default_inputs(self):
    model_type = self.user_config.clarifai_model.type
    if model_type == ModelTypes.multimodal_embedder:
      self.predict(input_data=[{IMAGE_TENSOR_NAME: each} for each in _default_images])
      self.predict(input_data=[{TEXT_TENSOR_NAME: each} for each in _default_texts])
      self.predict(input_data=[{
          TEXT_TENSOR_NAME: text,
          IMAGE_TENSOR_NAME: img
      } for text, img in zip(_default_texts, _default_images)])
    elif model_type.startswith("visual"):
      self.predict(input_data=_default_images)
    else:
      self.predict(input_data=_default_texts)
