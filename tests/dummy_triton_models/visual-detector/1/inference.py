# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union

import numpy as np

from clarifai.models.model_serving.model_config import VisualDetector, VisualDetectorOutput


class InferenceModel(VisualDetector):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int]]) -> list:
    """ Custom prediction function for `visual-detector` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_parameters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of VisualDetectorOutput

    """

    outputs = []

    for inp in input_data:
      assert isinstance(inp, np.ndarray), "Incorrect type of image, expected np.ndarray"
      bboxes = np.random.rand(1, 4)
      classes = np.random.randint(0, 1, size=(1, 1))
      scores = np.random.rand(1, 1)
      output = VisualDetectorOutput(
          predicted_bboxes=bboxes, predicted_labels=classes, predicted_scores=scores)
      outputs.append(output)

    return outputs
